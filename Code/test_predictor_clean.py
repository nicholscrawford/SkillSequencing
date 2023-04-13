import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point32
from pickle import load
from std_msgs.msg import ColorRGBA
import struct, threading
import torch
import os
import numpy as np
import lightning.pytorch as pl
from precondition_prediction_clean import LitPrecondPredictor, load_latest_ckpt


publishers = []
messages = []

# This function publishes point clouds to a ROS topic using the PointCloud2 message type. 
# It sets up a ROS publisher, defines a list of colors for each point cloud, and publishes the point clouds in a loop until the ROS node is shut down.
def display_scene_pointclouds(pointclouds):
    pub = rospy.Publisher('scene_pointclouds', PointCloud2, queue_size=10)
    publishers.append(pub)

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (0, 1, 1) ] #list of colors for each point cloud 

    
    # Create the PointCloud2 message
    pointcloud = PointCloud2()
    pointcloud.header.frame_id = "world"
    pointcloud.height = 1
    num_points = sum([len(points) for points in pointclouds])
    pointcloud.width = num_points

    # Define the fields for the pointcloud message
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1),
    ]
    pointcloud.fields = fields
    pointcloud.is_bigendian = False
    pointcloud.point_step = 24
    pointcloud.row_step = pointcloud.point_step * pointcloud.width

    # Create the data array for the pointcloud message
    data = []
    for i, points in enumerate(pointclouds):
        r, g, b = colors[i]
        for point in points:
            z, x, y = point
            data += struct.pack('f', x)
            data += struct.pack('f', y)
            data += struct.pack('f', z)
            data += struct.pack('f', r)
            data += struct.pack('f', g)
            data += struct.pack('f', b)
    pointcloud.data = data

    messages.append(pointcloud)
    

def display_intensity_pointcloud(points):
    # Create a publisher for the point cloud
    pub = rospy.Publisher("intensity_points", PointCloud2, queue_size=1)
    publishers.append(pub)

    # Create an empty point cloud message
    cloud_msg = PointCloud2()
    # Set the header information
    cloud_msg.header.stamp = rospy.Time.now()
    cloud_msg.header.frame_id = "world"
    # Set the point cloud fields
    cloud_msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]
    # Set the point cloud width and height
    cloud_msg.width = len(points)
    cloud_msg.height = 1
    # Set the point cloud data
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 16
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

    data = []
    for point in points:
        x, y, z, i = point
        data += struct.pack('f', x)
        data += struct.pack('f', y)
        data += struct.pack('f', z)
        data += struct.pack('f', i)
    cloud_msg.data = data

    
    messages.append(cloud_msg)

#Loads model, and runs it on even intervals and uses predicted success as pc intensity.
def create_param_viz(pcs, gt_param=None, gt_succ=None):
    #Implicit in skill
    x = 0.95
    z = 1.15
    
    model = load_latest_ckpt()

    with torch.no_grad():
        #Get encoded point cloud
        pcl = [pc for pc in list(pcs.values())]
        pcl = np.array(pcl)
        pc_tensor = torch.tensor(pcl, dtype=torch.float32, device=model.device)
        pc_tensor = pc_tensor.transpose(1, 2)

        num_points = 200
        points = []

        action_params = torch.tensor([
            [(idx/num_points )*0.6 - 0.3 for idx in range(num_points)],
            [(idx/num_points )*0.6 - 0.3 for idx in range(num_points)]
        ], device=model.device, dtype=torch.float32)
        successes = model((pc_tensor, action_params))
                #pc_encoding_param = torch.concat((pc_encoding, action_param), dim=0)
                #succ_hat = float(predictor(pc_encoding_param))
                #succ_hat = 0 if y > 0.1 or y < -0.1 else 1

        #points.append((x, y, z, succ_hat))
        points = [(x, (idx/num_points )*0.6 - 0.3, z, successes[0][idx].item()) for idx in range(num_points)] + \
            [(x, (idx/num_points )*0.6 - 0.3, z+0.03, successes[1][idx].item()) for idx in range(num_points)]
            
        
        if gt_param is not None:
            points.append((x, gt_param, z, gt_succ))

        
        points.append((x, -2, z, 1)) #Add min and max points for proper coloring in RVIZ
        points.append((x, -2, z, 0)) #Add min and max points for proper coloring in RVIZ

    return points

if __name__ == "__main__":
    io_pairs_filename = os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor', '2023-feb-8', 'TipThenPull', 'io_pairs.pickle')
    with open(io_pairs_filename, 'rb') as io_pairs_fd:
        io_pairs = load(io_pairs_fd)
        
        object_pcs, param, succ = io_pairs[np.random.randint(0, len(io_pairs)-1)]
        print(f"This example had a success value of {succ} with a parameter of {param}")

        rospy.init_node('pointcloud_display')
        
        display_scene_pointclouds([object_pcs['shelf'], object_pcs['box'], object_pcs['box2']])
        display_intensity_pointcloud(create_param_viz(object_pcs))#, gt_param= param, gt_succ=succ))
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            for i, pub in enumerate(publishers):
                pub.publish(messages[i])
                rate.sleep()
