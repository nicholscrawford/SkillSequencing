# Skill Sequencing

Repo to store code to for skill sequencing research.

Exploring using precondition prediction functions to get action parameters through sampling. Using simple transformations as skill effects models. 

## Usage Notes

### Planner
For demoing the planner, some backend must be set up. Moveit/Roscore must be spun up using ```roslaunch moveit_interface iiwa_reflex_moveit_interface_service.launch use_rviz:=True```. Current sim testing uses (in the ll4ma gym repo) the Gym Publisher script to spin up an instance of gym simulation as an analog for the real world, ``` $CATKIN_WS/src/ll4ma_isaac/ll4ma_isaacgym/src/ll4ma_isaacgym/scripts/isaacgym_publisher.py --fake```.

### Training Models

Training is currently done by running main, using the loaded model from precondition_prediction*.py. 

#### Quick Testing

Working on a quick testing module to do some visualizations, and have a more rapid turnaround time with model parameter/structure selection. That'll be in test_predictor.py.

### Collecting Data
I'm not sure exactly how I'll be adding more behaviors.

Currently, data is collecting using isaacgym scripts, refer to the ll4ma gym repo for more information (will note more required detail once I redo this/start adding more actions). That data is collected as a bunch of all-encompassing pickle files, stored as demo_####.pickle. Main will convert that to a more usable format, sampled, and combined. The format is currently 
```
{
    action_name: 
        [
            (pc, param, success), 
            (pc, param, success), 
            ...
        ],

    action_name: 
        [
            (pc, param, success), 
            (pc, param, success), 
            ...
        ] 
}

```
###

## Plan

Currently some things are implemented, but need refinement and cleaning. Current sampling doens't seem to be working. 

**TODO:** 
- Get sampling/prediction working
    - BCE Loss + PointConv seems to decrease loss rapidly 157-0.0x in >10 itter. Converges to 1 for all vals.
    - Try adding back in non-correct guesses for pickup?