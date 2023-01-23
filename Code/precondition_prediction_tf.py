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

    data = expand_data(data)

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
                
                pc_encoding = encoder(pc)

                #Add target and environment tags, pos 0 and 1
                ids = torch.tensor([[1, 0], [0,0], [0, 1]], device=device)
                pc_encoding = torch.concat((pc_encoding, ids), dim=1)
                
                #Add action parameter
                # pc_encoding = torch.reshape(pc_encoding, [-1])
                action_param = torch.tensor([param], device=device)
                # pc_encoding = torch.concat((pc_encoding, action_param), dim=0)

                succ_hat = predictor(pc_encoding, action_param)

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
        

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Predicts precondition from action param and encoded PC.
class Precond_Predictor(nn.Module):
    def __init__(self, action_param_size) -> None:
        super(Precond_Predictor, self).__init__()
        # self.action_param_size = action_param_size
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(3*(16+2)+action_param_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )
        self.self_attn = nn.MultiheadAttention(16+2, 9)
        self.fc = nn.Linear(18, 18)
        self.relu = nn.ReLU()
        self.self_attn2 = nn.MultiheadAttention(16+2, 9)
        self.fc2 = nn.Linear(18*3+1, 1)
        self.ln = LayerNorm((3, 18))

    def forward(self, x, ap):
        a, _ = self.self_attn(x,x,x) #Mask?
        x = a + x
        x = self.ln(x)
        x = self.fc(x)
        a, _ = self.self_attn2(x,x,x) #Mask?
        x = a + x
        x = self.ln(x)
        #Concat action param

        x = torch.reshape(x, [-1])
        x = torch.concat((x, ap), dim=0)

        x = self.fc2(x)
        return x