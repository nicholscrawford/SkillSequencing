import pickle
import os
import numpy as np
import random
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from farthest_point_sampling import farthest_point_sampling
from copy import deepcopy
from pickle import load

#Loads the data dictionary
def get_iopairs_dict(paths, resample = False):

    iopairs_dict = {}
    for path in paths:
        iopairs_path = os.path.join(path, "io_pairs.pickle")

        if (not os.path.exists(iopairs_path)) or resample:
            iopairs = get_pcparam_success_pairs(path)
            with open(iopairs_path, 'wb') as fd:
                pickle.dump(iopairs, fd, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"IO Pairs saved to {path.split('/')[-1]}.")
        else:
            with open(iopairs_path, 'rb') as fd:
                iopairs = pickle.load(fd)
                print(f"IO Pairs loaded from {path}")
        
        iopairs_dict[path.split('/')[-1]] = iopairs

    return iopairs_dict

# Load data from pickle files
def load_files(path = '/home/nichols/Desktop/SkillSequnceing/Data/Nov17/TipThenPull'):
    files = os.listdir(path)
    files = [file for file in files if os.path.splitext(file)[1] == '.pickle']

    data = []
    for file in files:
        with open(os.path.join(path, file), 'rb') as fd:
            data.append(pickle.load(fd))

    return data

# Transform pickle data to pc-parameters-success
def get_pcparam_success_pairs(path = '/home/nichols/Desktop/SkillSequnceing/Data/Nov17/TipThenPull'):
    data = load_files(path)
    pairs = [] # Eements of pc, param, success
    for dat in tqdm(data, "Loading pairs", len(data)):
        succ = get_success_from_gym(dat[0])
        param = get_action_params_from_gym(dat[0])
        pc = get_pc_from_gym(dat[0])
        sample_pointcloud(pc)

        pairs.append((pc['point_clouds'][0], param, succ))

    return pairs

# Sample from point clouds
def sample_pointcloud(data: dict):
    #Sampling number should probably be configurable, especially since it is needed by model. Alternatively, model could just
    #read what it is from data.
    sampling_number = 128

    for timestep_idx in range(len(data["depth"])):
        for object_name in data["objects"]:
            pc = data["point_clouds"][timestep_idx][object_name]
            if pc.ndim != 2:
                print(f"Pointcloud ndim {pc.ndim}, shape {pc.shape}, for object {object_name} in timestep {timestep_idx} of {len( data['depth'] )}.")
                continue
            farthest_indices,_ = farthest_point_sampling(pc, sampling_number)
            sampled_pc = pc[farthest_indices.squeeze()]
            data["point_clouds"][timestep_idx][object_name] = sampled_pc

def farthest_point_sampling_from_pc(pointcloud):
    sampling_number = 128
    for object_name in pointcloud.keys():
        pc = pointcloud[object_name]
        farthest_indices,_ = farthest_point_sampling(pc, sampling_number)
        sampled_pc = pc[farthest_indices.squeeze()]
        pointcloud[object_name] = sampled_pc
    return pointcloud


# Extract point clouds from data
def get_pc_from_gym(data):
    data["point_clouds"] = []

    for timestep_idx in range(len(data["depth"])):
        points = []

        data["point_clouds"].append({})

        for object_name in data["objects"].keys():
            data["point_clouds"][timestep_idx][object_name] = []

        color = []
        for i_o in range(len(data["objects"])):
            points.append([])
        cam_width = 512
        cam_height = 512

        # Retrieve depth and segmentation buffer
        depth_buffer = data["depth"][timestep_idx]
        seg_buffer = data["segmentation"][timestep_idx]
        view_matrix = data["view_matrix"][0]
        projection_matrix = data["projection_matrix"][0]

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                for i_o in range(len(data["objects"])):
                    #Assumes object order corresponds to their seg number, which I think is true, but could be good to confirm.
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
        for i_o in range(len(points)):
            points[i_o] = np.array(points[i_o])
        
        idx = 0
        for object_name in data["objects"].keys():
            data["point_clouds"][timestep_idx][object_name] = points[idx]
            idx += 1

    return data

# Get point cloud from rgb-d and camera and seg info.
def get_pc_from_rgb_d(rgb, dep, object_names, seg_img, seg_ids, proj_mat, view_mat):
    point_clouds = {}
    points = []

    for object_name in object_names:
        point_clouds[object_name] = []

        color = []
        for i_o in range(len(object_names)):
            points.append([])
        cam_width = 512
        cam_height = 512

        # Retrieve depth and segmentation buffer
        depth_buffer = np.array(dep)
        seg_buffer = seg_img
        view_matrix = view_mat
        projection_matrix = proj_mat

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                for i_o in range(len(object_names)):
                    #Assumes object order corresponds to their seg number, which I think is true, but could be good to confirm.
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
        
    for i_o in range(len(points)):
        points[i_o] = np.array(points[i_o])
        
    idx = 0
    for object_name in object_names:
        point_clouds[object_name] = points[idx]
        idx += 1

    return point_clouds

# Extract success metric from data
def get_success_from_gym(data):
    target_end_pos = data['objects']['box']['position'][-1]
    non_target_pos = data['objects']['box2']['position']
    
    #Check if box is still relatively high, hasn't fallen to floor.
    if target_end_pos[2] < 0.5:
        return False
    #Check if box has been pulled from shelf or not.
    if target_end_pos[0] > 1:
        return False

    #Check if non-target has been moved
    start = np.array(non_target_pos[0])
    end = np.array(non_target_pos[-1])
    dist = np.linalg.norm(start - end)
    if dist > 0.1:
        return False

    return True

# Extract parameters from data
def get_action_params_from_gym(data):
    return data['objects']['box']['position'][0][1]

# Add unrealistic tasks with failure.
def expand_data(data, factor = 3):
    for key in data.keys():
        for element in deepcopy(data[key]):
            for i in range(factor):
                peturbation = deepcopy(data[key][i][1])
                peturbation = random.choice([ random.uniform(-0.3, peturbation-0.05), random.uniform(peturbation+0.05, 0.3)])
                data[key].append(
                    [
                        deepcopy(data[key][i][0]), # Copy pc
                        peturbation, #Move arm to unrealistic position
                        torch.tensor(0, device = data[key][i][0].device) #Fail
                    ]
                )

    return data

# Add realistic tasks with success.
def expand_data_success(data, factor = 3):
    for key in data.keys():
        for element in deepcopy(data[key]):
            for i in range(factor):
                peturbation = deepcopy(data[key][i][1])
                peturbation = random.uniform(peturbation-0.05, peturbation+0.05)
                data[key].append(
                    [
                        deepcopy(data[key][i][0]), # Copy pc
                        peturbation, #Move arm to realistic position
                        torch.tensor(1, device = data[key][i][0].device, dtype=torch.float32) #Fail
                    ]
                )

    return data

#Plots a point cloud, doesn't really have good performance.
def show_pc(pointclouds):

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d') 
    
    color_arr = ['red', 'green', 'blue']
    for i, pointcloud in enumerate(pointclouds.values()):
        for point in pointcloud:
            if random.randint(0, 0) == 0:
                ax.scatter(point[0],-point[2],point[1],c=color_arr[i]) # plot the point (2,3,4) on the figure   
    
    # ax.set_xlim(0.6,1.4)
    # ax.set_ylim(-0.5,0.5)
    # ax.set_zlim(0.6,1.5)

    plt.show()

    
if __name__ == "__main__":
    io_pairs_filename = '/home/nichols/Desktop/SkillSequnceing/Data/Nov20/PullFromShelf/io_pairs.pickle'
    with open(io_pairs_filename, 'rb') as io_pairs_fd:
        io_pairs = load(io_pairs_fd)
        
        total = 0
        totalsucc = 0
        for i in range(len(io_pairs)):
            object_pcs, param, succ = io_pairs[i]
            total += 1
            if succ:
                totalsucc += 1
        
        print(f"Total success rate of {totalsucc/total} for PullFromShelf, with {total} samples.")

    io_pairs_filename = '/home/nichols/Desktop/SkillSequnceing/Data/Nov20/TipThenPull/io_pairs.pickle'
    with open(io_pairs_filename, 'rb') as io_pairs_fd:
        io_pairs = load(io_pairs_fd)
        
        total = 0
        totalsucc = 0
        for i in range(len(io_pairs)):
            object_pcs, param, succ = io_pairs[i]
            total += 1
            if succ:
                totalsucc += 1
        
        print(f"Total success rate of {totalsucc/total} for TipThenPull, with {total} samples.")