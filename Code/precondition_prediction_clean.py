import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from pointconvds import PointConvDensitySetAbstraction
from action_info import action
import torch
import numpy as np

import lightning.pytorch as pl

class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5


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
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)

        return x

class Precond_Predictor(nn.Module):
    def __init__(self, action_param_size) -> None:
        super(Precond_Predictor, self).__init__()
        self.action_param_size = action_param_size
        self.linear_relu_stack = nn.Sequential(

            nn.Linear(3*(128+2)+action_param_size, 128),
            nn.GroupNorm(1, 128),
            #nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.GroupNorm(1, 128),
            #nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        logits = self.linear_relu_stack(x)
        return logits
    
# define the LightningModule
class LitPrecondPredictor(pl.LightningModule):

    def __init__(self, actions, lr) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.predictors = nn.ModuleList([Precond_Predictor(action.action_num_params) for action in actions])
        self.encoder = PointConv()
        self.actions = actions
        self.loss_func = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        '''
        Run a bunch of parameters per action on a point cloud. 
        We run each objects point cloud through pointconv as a batch, 
        then duplicate that encoding per-parameter per-action and run a batch for each action.
        '''

        pointcloud, params = x
        predictions = []

        #Change shape for encoder
        #action_pointcloud = action_pointcloud.transpose(1, 2)
        
        pointcloud_encoding = self.encoder(pointcloud)
        
        #Add target and environment tags, pos 0 and 1
        ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=self.device)
        pointcloud_encoding = torch.concat((pointcloud_encoding, ids), dim=1)
        pointcloud_encoding = torch.reshape(pointcloud_encoding, [-1]) #Flatten

        # expand pointcloud_encoding to shape [1, 390] to allow broadcasting
        pointcloud_encoding = pointcloud_encoding.unsqueeze(0)


        #Run a pass for each action
        for action in self.actions:
            predictor = self.predictors[action.action_module_idx]
            
            # repeat pointcloud_encoding along the first dimension to create two tensors
            pointcloud_encoding_batch = pointcloud_encoding.repeat(params.shape[1], 1)

            action_params = params[action.action_module_idx].unsqueeze(-1)

            #Add action parameter 
            encoding_and_params = torch.concat((pointcloud_encoding_batch, action_params), dim=1) #Combine
                       

            succ_hat = predictor(encoding_and_params)
            predictions.append(succ_hat)

        return predictions

    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("validation_loss", loss, sync_dist=True)
    
    def testing_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, sync_dist=True)

    def run_batch(self, batch, batch_idx):
        '''
        Run a batch, for training and validation. 
        Put here so the training and validation code isn't duplicated.
        '''

        (pointclouds, params), success_gts = batch
        loss_sum = 0
        
        #Run a pass for each action
        for action in self.actions:
            predictor = self.predictors[action.action_module_idx]
            action_pointclouds = pointclouds[action.action_module_idx]
            action_params = params[action.action_module_idx]
            action_success_gts = success_gts[action.action_module_idx]

            #Run each pc in batch through the encoder seperately
            pointcloud_encodings = []
            for intrabatch_idx, pointcloud in enumerate(action_pointclouds):
                #Change shape for encoder
                pointcloud = pointcloud.transpose(1, 2)
                
                # Add gaussian noise.
                pointcloud = pointcloud + torch.tensor(np.random.normal(0, 0.001, tuple(pointcloud.shape)), device=pointcloud.device, dtype=pointcloud.dtype) 
                
                pointcloud_encoding = self.encoder(pointcloud)
                
                #Add target and environment tags, pos 0 and 1
                ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=self.device)
                pointcloud_encoding = torch.concat((pointcloud_encoding, ids), dim=1)

                #Add action parameter for this example
                pointcloud_encoding = torch.reshape(pointcloud_encoding, [-1]) #Flatten
                param = action_params[intrabatch_idx].unsqueeze(0)
                #param = param + np.random.uniform(-0.03, 0.03) #Add uniform noise to action param
                param = param + np.random.normal(0, 0.005) #Add gaussian noise to action param
                pointcloud_encoding = torch.concat((pointcloud_encoding, param), dim=0) #Combine

                pointcloud_encodings.append(pointcloud_encoding)
            
            #Put them back together
            pointcloud_encodings = torch.stack(pointcloud_encodings)            

            succ_hats = predictor(pointcloud_encodings)

            action_success_gts = torch.stack([action_success_gts]).T
         
            loss_out = self.loss_func(succ_hats, action_success_gts)
            loss_sum += loss_out

        return loss_sum
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

