from typing import List
from dataclasses import dataclass, asdict

@dataclass
class io_pairs:
    point_cloud: list
    parameters: list
    success: bool

@dataclass
class action_data:
    action_name: str
    actions_data: List[io_pairs]

@dataclass
class train_data:
    action_pairs: List[action_data]