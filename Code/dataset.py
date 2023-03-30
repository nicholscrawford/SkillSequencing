from dataclasses import dataclass
import enum
from re import T
from torch.utils.data import DataLoader
from action_info import action
import torch
import numpy as np
from copy import deepcopy

import random


def get_dataloaders_dict(task_io_pairs, batch_size):
    dataloaders = {}
    for task, pairs in task_io_pairs.items():
        dataloaders[task] = DataLoader(task_dataset(pairs), batch_size=batch_size, shuffle=True)

    return dataloaders



def get_list_dataloader(iopairs, actions, batch_size):
    if type(iopairs[0][0][0]) == torch.Tensor:
        return DataLoader(list_task_dataset(iopairs, actions), batch_size=batch_size, shuffle=False)
    
    for action_iopairs_idx, action_iopairs in enumerate(iopairs):
                # Format data to tensors
            for datum_idx, datum in enumerate(action_iopairs):
                pcl = [pc for pc in list(datum[0].values())]
                pcl = np.array(pcl)
                iopairs[action_iopairs_idx][datum_idx] = [
                    torch.tensor(pcl, dtype=torch.float32),
                    torch.tensor(datum[1], dtype=torch.float32), #action param
                    torch.tensor(datum[2], dtype=torch.float32),
                    ]


    return DataLoader(list_task_dataset(iopairs, actions), batch_size=batch_size, shuffle=False)


class task_dataset:
    def __init__(self, io_pairs) -> None:
        self.io_pairs = io_pairs

    def __len__(self) -> int:
        return len(self.io_pairs)
    
    def __getitem__(self, idx):
        point_cloud = self.io_pairs[idx][0]
        param = self.io_pairs[idx][1]
        success = self.io_pairs[idx][2]
        return (point_cloud, param), success
    


    
class list_task_dataset:
    def __init__(self, io_pairs_list, actions) -> None:
        self.io_pairs_list = io_pairs_list
        self.actions = actions

    def __len__(self) -> int:
        return len(self.io_pairs_list[0])
    
    def __getitem__(self, idx):
        #Shape is [num_actions]
        point_clouds = [self.io_pairs_list[action.action_module_idx][idx][0] for action in self.actions]
        params = [self.io_pairs_list[action.action_module_idx][idx][1] for action in self.actions]
        success = [self.io_pairs_list[action.action_module_idx][idx][2] for action in self.actions]
        return (point_clouds, params), success
    
    def expand(self, expand_factor = 9):
        for action_idx, elements in enumerate(self.io_pairs_list):
            for i in range(len(elements)):
                for _ in range(expand_factor):
                    peturbation = deepcopy(elements[i][1])
                    peturbation = random.choice([ random.uniform(-0.3, peturbation-0.05), random.uniform(peturbation+0.05, 0.3)])

                    self.io_pairs_list[action_idx].append(
                        [
                            deepcopy(elements[i][0]), # Copy pc
                            peturbation, #Move arm to unrealistic position
                            torch.tensor(0, dtype=torch.float32) #Fail
                        ]
                    )

                    peturbation = deepcopy(elements[i][1])
                    peturbation = random.uniform(peturbation-0.05, peturbation+0.05)
                    self.io_pairs_list[action_idx].append(
                        [
                            deepcopy(elements[i][0]), # Copy pc
                            peturbation, #Move arm to realistic position
                            torch.tensor(1, dtype=torch.float32) #Fail
                        ]
                    )