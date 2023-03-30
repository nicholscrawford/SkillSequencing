from typing import List
from dataclasses import dataclass, asdict

@dataclass
class action:
    action_name: str
    action_module_idx: int
    action_num_params: int
