import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from tqdm import tqdm
import sys
import os

# Predict preconditions from a shared latent embedding. 

# Training method given a dict of lists of tensor tuples, so a list of datapoints, where each point is (pc, params, success).
# So, you can access using each predictor's name.
def train(encoder, list_of_predictors, data, epochs=200, device=torch.cuda.is_available()):

    #Create optimizers
    lr = 0.001
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr = lr)
    predictor_opts = {}
    for task, predictor in list_of_predictors.items():
        predictor_opts[task] = torch.optim.Adam(predictor.parameters(), lr = lr)

    #Define training step
    print("Training model:")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(data[list(data.keys())[0]])):
            
            encoder_opt.zero_grad()
            for task, predictor in list_of_predictors.items():
                pc, param, succ = data[task][i]

                predictor_opts[task].zero_grad()
                
                pc_encoding = encoder(pc)

                #Add target and environment tags, pos 0 and 1
                ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=device)
                pc_encoding = torch.concat((pc_encoding, ids), dim=1)
                
                #Add action parameter
                pc_encoding = torch.reshape(pc_encoding, [-1])
                action_param = torch.tensor([param], device=device)
                pc_encoding = torch.concat((pc_encoding, action_param), dim=0)

                succ_hat = predictor(pc_encoding)

                loss = ((succ - succ_hat)**2).sum()
                loss.backward()
                total_loss += loss
                
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
class PC_Encoder(nn.Module):
    def __init__(self):
        super(PC_Encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        

# Predicts precondition from action param and encoded PC.
class Precond_Predictor(nn.Module):
    def __init__(self, action_param_size) -> None:
        super(Precond_Predictor, self).__init__()
        self.action_param_size = action_param_size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*(16+2)+action_param_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits