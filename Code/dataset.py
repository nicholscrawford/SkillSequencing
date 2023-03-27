from dataclasses import dataclass
from re import T
from torch.utils.data import DataLoader
from action_info import action




def get_dataloaders_dict(task_io_pairs, batch_size):
    dataloaders = {}
    for task, pairs in task_io_pairs.items():
        dataloaders[task] = DataLoader(task_dataset(pairs), batch_size=batch_size, shuffle=True)

    return dataloaders

def get_list_dataloader(io_pairs_list, actions, batch_size):
    return DataLoader(task_dataset(io_pairs_list, actions), batch_size=batch_size, shuffle=True)


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
        self.actions

    def __len__(self) -> int:
        return len(self.io_pairs[0])
    
    def __getitem__(self, idx):
        #Shape is [num_actions]
        point_clouds = [self.io_pairs_list[action.action_module_idx][idx][0] for action in self.actions]
        params = [self.io_pairs_list[action.action_module_idx][idx][1] for action in self.actions]
        success = [self.io_pairs_list[action.action_module_idx][idx][2] for action in self.actions]
        return (point_clouds, params), success