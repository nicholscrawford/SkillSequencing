from dataclasses import dataclass
from re import T
from torch.utils.data import DataLoader




def get_dataloaders(task_io_pairs, batch_size):
    dataloaders = {}
    for task, pairs in task_io_pairs.items():
        dataloaders[task] = DataLoader(task_dataset(pairs), batch_size=batch_size, shuffle=True)

    return dataloaders

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