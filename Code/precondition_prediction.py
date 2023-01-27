from data_utils import expand_data
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from tqdm import tqdm
import sys
import random
import os
from pointconvds import PointConvDensitySetAbstraction

# Predict preconditions from a shared latent embedding. 

# Training method given a dict of lists of tensor tuples, so a list of datapoints, where each point is (pc, params, success).
# So, you can access using each predictor's name.
def train(encoder, list_of_predictors, data, epochs=20, device=torch.cuda.is_available()):

    #Create optimizers
    lr = 0.0001
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr = lr)
    predictor_opts = {}
    for task, predictor in list_of_predictors.items():
        predictor_opts[task] = torch.optim.Adam(predictor.parameters(), lr = lr)

    data = expand_data(data)
    loss = nn.BCELoss()

    #Define training step
    print("Training model:")
    for epoch in range(epochs):
        total_loss = 0
        for idx in range(len(data[list(data.keys())[0]])):
            i = random.randint(0, len(data[list(data.keys())[0]])-1)
            
            for task, predictor in list_of_predictors.items():
                encoder_opt.zero_grad()
                pc, param, succ = data[task][i]

                predictor_opts[task].zero_grad()
                
                #Change shape for encoder
                pc = pc.transpose(1, 2)
                
                pc_encoding = encoder(pc)

                #Add target and environment tags, pos 0 and 1
                ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=device)
                pc_encoding = torch.concat((pc_encoding, ids), dim=1)
                
                #Add action parameter
                pc_encoding = torch.reshape(pc_encoding, [-1]) #Flatten
                action_param = torch.tensor([param], device=device) #Tensorify param
                pc_encoding = torch.concat((pc_encoding, action_param), dim=0) #Combine

                succ_hat = predictor(pc_encoding)

                #loss = ((succ - succ_hat)**2).sum()
                succ = torch.tensor([succ], device=succ_hat.device)
                loss_out = loss(succ_hat, succ_hat)
                loss_out.backward()
                total_loss += loss_out
                
                predictor_opts[task].step()
                encoder_opt.step()

        sys.stdout.write(f"\rTotal loss at epoch {epoch+1}/{epochs} is {total_loss}")
        if((epoch+1)%(int(epochs/5 + 0.5))==0):
            # Save models.
            # It's probably fine to just save in f"Checkpoints/ModelName_{epoch}of{epochs}.pth"
            codepath = os.path.join(os.getcwd(), os.path.dirname(__file__))
            with open(os.path.join(codepath, f"Checkpoints/Encoder_{epoch+1}of{epochs}.pth"), "wb") as fd:
                torch.save(encoder, fd)
            for task, predictor in list_of_predictors.items():
                with open(os.path.join(codepath, f"Checkpoints/Predictor{task}_{epoch+1}of{epochs}.pth"), "wb") as fd:
                    torch.save(predictor, fd)
            print()
    print()




# Encodes point clouds to a shared latent space
# class PC_Encoder(nn.Module):
#     def __init__(self):
#         super(PC_Encoder, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(128*3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 16),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# Predicts precondition from action param and encoded PC.
class Precond_Predictor(nn.Module):
    def __init__(self, action_param_size) -> None:
        super(Precond_Predictor, self).__init__()
        self.action_param_size = action_param_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*(128+2)+action_param_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[512], bandwidth = 0.4, group_all=True)
        
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5

        # self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[16], bandwidth = 0.2, group_all=True) # version feb
        
        
        #self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[128], bandwidth = 0.4, group_all=True)  

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))



        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x

class PC_Encoder(PointConv):
    pass
