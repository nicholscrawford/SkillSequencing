#!/usr/bin/env python
"""
This script is intended to be used to plan actions using isaacgym_publisher. It was based off of 
behavior_runner in the ll4ma_isaacgym repository. 

This script takes a configuration file as input and uses the task parameters in
order to create the behavior tree. It then gets the environment state from
/iiwa/envstate (either a fake one from isaacgym_publisher.py or a real one TBD
that estimates object states from the pointcloud/rgbd info). It uses the envstate
to determine the plan and then executes that plan on the real robot.
"""
from isaacgym import gymtorch, gymapi  # noqa: F401
import sys
import argparse
from ll4ma_isaacgym.core.config import PullFromShelfConfig, TipThenPullConfig
import rospy
import tf
import json
import os
import torch
from cv_bridge import CvBridge
import numpy as np

from matplotlib import pyplot as plt
from std_srvs.srv import Trigger, TriggerRequest
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory
from ll4ma_isaacgym.srv import EstState, EstStateRequest

from ll4ma_util import file_util, ros_util

from ll4ma_isaacgym.core import SessionConfig, EnvironmentState
from ll4ma_isaacgym.robots import Robot
from ll4ma_isaacgym.behaviors import Behavior
from ll4ma_isaacgym.msg import IsaacGymState

from moveit_interface.iiwa_planner import IiwaPlanner
from reflex import ReflexGraspInterface

from precondition_prediction import PC_Encoder, Precond_Predictor
from data_utils import get_pc_from_rgb_d, farthest_point_sampling_from_pc
from sampling import sample_for_action_param

class BehaviorRunner:
    """docstring for BehaviorRunner."""

    def __init__(self, session_config, rate, use_reflex, taskcfgdict, fake=False):
        self.session_config = session_config
        self.use_reflex = use_reflex
        self.taskcfgdict = taskcfgdict

        self.rate = rospy.Rate(rate)
        self.tf_listener = tf.TransformListener()
        # This is in IiwaPlanner:
        # self.cmd_pub = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=1)
        rospy.Subscriber("/iiwa/joint_states", JointState, self.joint_state_cb)

        self.joint_state = None
        self.obj_state = None

        self.robot_config = next(iter(session_config.env.robots.values()))
        self.robot = Robot(self.robot_config)
        self.n_arm_joints = self.robot.arm.num_joints()
        self.n_ee_joints = self.robot.end_effector.num_joints()

        self.state = EnvironmentState()
        self.state.dt = 1.0 / rate
        self.state.n_arm_joints = self.n_arm_joints
        self.state.n_ee_joints = self.n_ee_joints
        self.state.objects = self.session_config.env.objects
        self.state.point_clouds = {}

        self.state.seg_img = None
        self.state.depth_img = None
        self.state.seg_ids = None
        self.state.proj_mat = None
        self.state.view_mat = None

        self.state.joint_names = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
            "reflex_preshape_joint_1",
            "reflex_proximal_joint_1",
            "reflex_preshape_joint_2",
            "reflex_proximal_joint_2",
            "reflex_proximal_joint_3",
        ]
        # self.state.joint_position = torch.zeros((self.n_arm_joints+self.n_ee_joints, 1))
        # self.state.joint_velocity = torch.zeros((self.n_arm_joints+self.n_ee_joints, 1))
        # self.state.joint_torque = torch.zeros((self.n_arm_joints+self.n_ee_joints, 1))

        self.use_state_service = False
        if not self.use_state_service:
            rospy.Subscriber("/iiwa/state_est/full", IsaacGymState, self.est_state_cb)
            self.est_state = IsaacGymState()
        else:
            self.est_state = EstState()
        self.est_state_filled = False

        self.fake = fake
        self.iiwa = IiwaPlanner(rate)
        if use_reflex:
            if self.fake:
                from reflex_msgs2.msg import Hand

                self.hand_state = None
                rospy.Subscriber("/reflex_takktile2/hand_state", Hand, self.hand_cb)
                self.reflex_pub = rospy.Publisher("/reflex_takktile2/action", String, queue_size=20)
                self.env_reset_pub = rospy.Publisher("/iiwa/isaacgym_reset", String, queue_size=20)
            else:
                self.reflex = ReflexGraspInterface()

        # Subscribe for seg_mat ids
        rospy.Subscriber("/isaacgym/seg_img", Image, self.seg_img_cb)
        rospy.Subscriber("/isaacgym/depth_img", Image, self.depth_img_cb)
        rospy.Subscriber("/isaacgym/seg_ids", String, self.seg_ids_cb)
        rospy.Subscriber("/isaacgym/proj_mat", Image, self.proj_mat_cb)
        rospy.Subscriber("/isaacgym/view_mat", Image, self.view_mat_cb)

        
        codepath = os.path.join(os.getcwd(), os.path.dirname(__file__))
        #Temporary shortcut for loading specific models. Eventually should be a param.
        self.list_of_predictors = {
            "PullFromShelf": None,
            "TipThenPull": None
        }
        epochs = 50
        epoch = 49

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(codepath, f"Checkpoints/Encoder_{epoch+1}of{epochs}.pth"), "rb") as fd:
            self.encoder = torch.load(fd)
            self.encoder.to(self.device)
        for task, predictor in self.list_of_predictors.items():
            with open(os.path.join(codepath, f"Checkpoints/Predictor{task}_{epoch+1}of{epochs}.pth"), "rb") as fd:
                predictor = torch.load(fd)
                predictor.to(self.device)
                self.list_of_predictors[task] = predictor
        print(f"Loaded models from epoch {epoch+1} out of of {epochs}.")


    def hand_cb(self, msg):
        self.hand_state = msg

    def get_est_state(self):
        resp, success = ros_util.call_service(EstStateRequest(), "/iiwa/get_state", EstState)
        if resp is not None and success:
            self.est_state = resp
            self.est_state_filled = True

    def reset_env(self):
        rospy.loginfo("Resetting environment...")
        resp, success = ros_util.call_service(TriggerRequest(), "/isaacgym/reset_env", Trigger)
        print(resp)
        import pdb

        pdb.set_trace()
        if not success:
            print(resp)
            import pdb

            pdb.set_trace()
        self.joint_state = None
        self.est_state_filled = False
        self.rate.sleep()
        rospy.loginfo("Environment reset")

    def reset(self, go_home=False):
        rospy.loginfo("Resetting behaviors...")
        self.behavior = Behavior(
            self.session_config.task.behavior,
            self.robot,
            self.session_config.env,
            None,
            self.session_config.device,
            open_loop=True,
        )
        if self.use_reflex:
            self.open_hand()
        if go_home:
            self.go_home()
        rospy.loginfo("Behavior reset")

    def go_home(self, update_obj_state=False, preview=True):
        objects = {}
        self.update_env_state(update_obj_state)
        traj = self.iiwa.get_plan("home", objects, vel_scaling=0.2)
        if traj is not None:
            self.command_trajectory(traj, preview=preview)
        else:
            rospy.logerr("Could not go home, got empty plan trajectory")

    def go_zero(self, update_obj_state=False, preview=True):
        objects = {}
        self.update_env_state(update_obj_state)
        traj = self.iiwa.get_plan("zero", objects, vel_scaling=0.2)
        if traj is not None:
            self.command_trajectory(traj, preview=preview)
        else:
            rospy.logerr("Could not go zero, got empty plan trajectory")

    def run_behavior(self, preview=True):
        if not self.update_env_state():
            rospy.logerr("State could not be fully updated")
            return False


        # This is probably where the planning should happen. Create self.behavior, set policy,  then send. 
        # First, create the current pc embedding.
        pcs = get_pc_from_rgb_d(
            rgb = self.state.rgb,
            dep = self.state.depth_img,
            object_names = self.state.objects,
            seg_img = self.state.seg_img,
            seg_ids = self.state.seg_ids,
            proj_mat = self.state.proj_mat,
            view_mat = self.state.view_mat
        )
        pcs: None = farthest_point_sampling_from_pc(pcs)
        pcl = [pc for pc in list(pcs.values())]
        np.array(pcl)
        pc_tensor = torch.tensor(pcl, dtype=torch.float32, device=self.device)
        pc_encoding = self.encoder(pc_tensor)
        #Add target and environment tags, pos 0 and 1
        ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=self.device)
        pc_encoding = torch.concat((pc_encoding, ids), dim=1)
        
        #Do sampling to find action param.
        pc_encoding = torch.reshape(pc_encoding, [-1])
        
        action_param, what_action = sample_for_action_param(pc_encoding, self.list_of_predictors) # Desired pc?

        #self.behavior.set_target... Apply action param according to behavior/action, maybe reinitialize behavior
        selected_behavior = self.taskcfgdict[what_action]
        selected_behavior.behaviors[0].action_param = action_param

        self.behavior = Behavior(
            #Behavior(TipThenPullConfig(), self.robot, self.session_config.env, None, self.session_config.device, open_loop=True).set_policy(self.state),
            selected_behavior,
            self.robot,
            self.session_config.env,
            None,
            self.session_config.device,
            open_loop=True,
        )

        if not self.behavior.set_policy(self.state):
            rospy.logerr("Could not get behavior trajectories")
            return False

        actions, names = self._flatten_ros_actions(self.behavior.get_ros_actions())
        for ai, action in enumerate(actions):
            if isinstance(action, JointTrajectory):
                self.command_trajectory(action, preview=preview, wait_secs_for_at_goal=1.5)
            elif isinstance(action, str):
                if action == "close":
                    if not self.use_reflex:
                        rospy.logerr(
                            "Cannot execute 'close' behavior, must set "
                            "use_reflex to true on class creation"
                        )
                        # sys.exit()
                    self.close_hand()
                elif action == "open":
                    if not self.use_reflex:
                        rospy.logerr(
                            "Cannot execute 'open' behavior, must set "
                            "use_reflex to true on class creation"
                        )
                        # sys.exit()
                    self.open_hand()
                elif action == "wide" or action == "close_lone" or action == "close_pair":
                    self.fake_hand(action)
                else:
                    rospy.logerr(f"Unknown action string: {action}")
            else:
                rospy.logerr(f"Unknown action type: {type(action)}")
                return

    def _flatten_ros_actions(self, traj):
        actions = []
        names = []
        for k, v in traj.items():
            if isinstance(v, dict):
                a, n = self._flatten_ros_actions(v)
                actions += a
                names += n
            else:
                actions.append(v)
                names.append(k)
        return actions, names

    def update_env_state(self, update_obj_state=True):
        if not self.wait_for_state_info():
            rospy.logerr("State info never completed")
            return False
        success = self.update_joint_state()
        success = success and self.update_ee_state()
        if update_obj_state:
            success = success and self.update_obj_state()
        success = success and self.update_rgbd_state()
        return success

    def update_joint_state(self):
        self.state.joint_position = torch.tensor(self.joint_state.position).unsqueeze(-1)
        self.state.joint_velocity = torch.tensor(self.joint_state.velocity).unsqueeze(-1)
        self.state.joint_torque = torch.tensor(self.joint_state.effort).unsqueeze(-1)
        self.state.joint_names = self.joint_state.name
        # TODO make this less hacked
        self.state.prev_action = torch.tensor(list(self.joint_state.position[:7]) + [0])
        return True

    def update_ee_state(self):
        rospy.loginfo("Getting EE pose...")
        trans, rot = self.lookup_tf("/world", "/reflex_palm_link")
        if trans is None or rot is None:
            rospy.logerr("Could not update EE pose")
            return False
        else:
            self.state.ee_state = torch.cat([torch.tensor(trans), torch.tensor(rot)])
            rospy.loginfo("EE pose found!")
            return True

    def update_obj_state(self):
        #print(self.est_state.object_poses)
        for oi, obj_name in enumerate(self.est_state.object_names):
            obj_name = obj_name.data
            if self.use_state_service:
                obj_state = self.est_state.object_poses[oi]
                self.state.object_states[obj_name] = torch.tensor(
                    [
                        obj_state.position.x,
                        obj_state.position.y,
                        obj_state.position.z,
                        obj_state.orientation.x,
                        obj_state.orientation.y,
                        obj_state.orientation.z,
                        obj_state.orientation.w,
                    ]
                )
            else:
                obj_state = self.est_state.objects[0].transforms[oi]
                assert obj_state.child_frame_id == obj_name
                self.state.object_states[obj_name] = torch.tensor(
                    [
                        obj_state.transform.translation.x,
                        obj_state.transform.translation.y,
                        obj_state.transform.translation.z,
                        obj_state.transform.rotation.x,
                        obj_state.transform.rotation.y,
                        obj_state.transform.rotation.z,
                        obj_state.transform.rotation.w,
                    ]
                )
        return True

    def update_rgbd_state(self) -> bool:
        self.est_state.rgb #List of rgb messages
        self.est_state.depth #List of depth messages
        
        self.state.rgb = np.frombuffer(self.est_state.rgb[-1].data, dtype=np.uint8).reshape(self.est_state.rgb[-1].height, self.est_state.rgb[-1].width, -1)
        self.state.depth = np.frombuffer(self.est_state.depth[-1].data, dtype=np.uint8).reshape(self.est_state.depth[-1].height, self.est_state.rgb[-1].width, -1)
        
        # Debugging for message interpretation.
        # plt.imshow(self.state.depth)
        # plt.show()
        return True #This is a hack. Need a better success measure, and to include the imgs not being accounted for here.


    def open_hand(self):
        if self.fake:
            self.fake_hand("open")
        else:
            self.reflex.open_hand()

    def close_hand(self):
        if self.fake:
            self.fake_hand("close")
        else:
            # TODO should have these on config
            close_velocity = 1.0
            max_close_position = [2.7] * 3
            use_tactile_stops = False
            use_velocity_stops = True
            tighten_increment = 0.3

            self.reflex.grasp(
                close_velocity,
                max_close_position,
                use_tactile_stops,
                use_velocity_stops,
                tighten_increment,
            )

    def fake_hand(self, action):
        for _ in range(10):
            self.reflex_pub.publish(String(action))
            self.rate.sleep()

    def est_state_cb(self, msg):
        self.est_state = msg
        self.est_state_filled = True

    def joint_state_cb(self, msg):
        self.joint_state = msg

    def seg_img_cb(self, msg):
        self.state.seg_img = CvBridge().imgmsg_to_cv2(msg)

    def depth_img_cb(self, msg):
        self.state.depth_img = CvBridge().imgmsg_to_cv2(msg)

    def seg_ids_cb(self, msg):
        self.state.seg_ids = json.loads(msg.data)

    def proj_mat_cb(self, msg):
        self.state.proj_mat = CvBridge().imgmsg_to_cv2(msg)

    def view_mat_cb(self, msg):
        self.state.view_mat = CvBridge().imgmsg_to_cv2(msg)


    def wait_for_state_info(self, timeout=30):
        if self.use_state_service:
            self.get_est_state()
        if self.joint_state is None or not self.est_state_filled:
            rospy.loginfo("Waiting for joint state...")
            start = rospy.get_time()
            while (
                not rospy.is_shutdown()
                and rospy.get_time() - start < timeout
                and (self.joint_state is None or not self.est_state_filled)
            ):
                self.rate.sleep()
            if self.joint_state is not None:
                rospy.loginfo("Joint state received")
            else:
                rospy.logerr("Joint state unknown")
                return False
            if self.est_state_filled:
                rospy.loginfo("Environment state received")
            else:
                rospy.logerr("Environment state unknown")
                return False
        return True

    def lookup_tf(self, from_frame, to_frame):
        trans = rot = None
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and trans is None:
            try:
                trans, rot = self.tf_listener.lookupTransform(from_frame, to_frame, rospy.Time())
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn(e)
            rate.sleep()
        return trans, rot

    def command_trajectory(
        self,
        traj,
        rate=100,
        at_goal_tolerance=0.01,
        wait_secs_for_at_goal=0,
        wait_secs_after_commanded=0,
        preview=True,
    ):
        self.iiwa.set_rate(rate)
        self.iiwa.command_trajectory(
            traj,
            False,
            at_goal_tolerance,
            wait_secs_for_at_goal,
            wait_secs_after_commanded,
            preview,
        )


if __name__ == "__main__":
    rospy.init_node("run_iiwa_behaviors", anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/home/nichols/catkin_ws/src/ll4ma_isaac/ll4ma_isaacgym/src/ll4ma_isaacgym/config/iiwa_tip_then_pull.yaml",
        help="Filename of YAML config for simulation (relative to current directory)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--use_reflex", action="store_true")
    parser.add_argument("--no_use_reflex", dest="use_reflex", action="store_false")
    parser.add_argument("--test_hand", action="store_true")
    parser.add_argument("--test_perception", action="store_true")
    parser.add_argument(
        "--reset", action="store_true", help="Set true to reset environment and exit"
    )
    parser.add_argument(
        "--go_home", action="store_true", help="Move robot to home configuration on reset"
    )
    parser.add_argument(
        "--no_go_home",
        dest="go_home",
        action="store_false",
        help="Leave robot in current configuration on reset",
    )
    parser.add_argument(
    "--primitive_task_configs", 
    nargs="*",
    type=str,
    default=[
        "TipThenPull:/home/nichols/catkin_ws/src/ll4ma_isaac/ll4ma_isaacgym/src/ll4ma_isaacgym/config/iiwa_tip_then_pull.yaml", 
        "PullFromShelf:/home/nichols/catkin_ws/src/ll4ma_isaac/ll4ma_isaacgym/src/ll4ma_isaacgym/config/iiwa_pull_from_shelf.yaml",
        ],
    )
    parser.add_argument("--fake", action="store_true")
    parser.set_defaults(fake=False)
    parser.set_defaults(use_reflex=True)
    parser.set_defaults(go_home=True)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    file_util.check_path_exists(args.config, "Config file")
    session_config = SessionConfig(config_filename=args.config)
    session_config.device = args.device
    taskcfgdict = {item.split(':')[0] : SessionConfig(config_filename=item.split(':')[1]).task.behavior for item in args.primitive_task_configs}
    runner = BehaviorRunner(session_config, args.rate, args.use_reflex, taskcfgdict, args.fake)

    # import pdb; pdb.set_trace()
    # runner.reflex.test()
    # import pdb; pdb.set_trace()

    if args.reset:
        runner.reset_env()

    runner.reset(False)
    try:
        runner.run_behavior(preview=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)

    # input('Continue?')
    runner.reset(False)
    runner.go_zero()
    rospy.loginfo("Behavior execution complete.")
