from ll4ma_isaacgym.core.config import TipThenPullConfig
from ll4ma_opt.problems import Problem 
from ll4ma_opt.solvers import CEM
from ll4ma_opt.solvers.samplers import GaussianSampler
from precondition_prediction import Precond_Predictor

import torch

def sample_for_action_param(model, pc_tensor, desired_pc):
    #task_param_dict = {k : 0 for k in list_of_predictors.keys()}
    #task_success_dict = {k : 0 for k in list_of_predictors.keys()}

    # n_samples = 200
    # n_elite = 10
    # max_iterations = 10

    # #Find optimal params per action
    # for predictor in list_of_predictors.keys():
    #     problem = FindActionParam(list_of_predictors[predictor], pc_encoding, 2)
    #     sampler = GaussianSampler(problem, start_iso_var=0.5)
    #     solver = CEM(problem, sampler, n_samples=n_samples, n_elite=n_elite)
    #     result = solver.optimize(max_iterations=max_iterations)
    #     task_param_dict[predictor] = result.solution[:1]
    #     task_success_dict[predictor] = result.cost

   
    # chosen_task = max(task_success_dict, key=task_success_dict.get)

    # return task_param_dict[chosen_task], chosen_task

    num_points = 2000
    points = []

    action_params = torch.tensor([
        [(idx/num_points )*0.6 - 0.3 for idx in range(num_points)],
        [(idx/num_points )*0.6 - 0.3 for idx in range(num_points)]
    ], device=model.device, dtype=torch.float32)
    successes = model((pc_tensor, action_params))

    selected_action_idx = torch.argmax(torch.tensor([max(action_successes) for action_successes in successes])).item()
    selected_action_parameter = action_params[selected_action_idx][torch.argmax(successes[selected_action_idx]).item()].item()
    return selected_action_idx, selected_action_parameter

    


class FindActionParam(Problem):
    def __init__(self, predictor, pc_encoding, size):
        self.param_size = size
        self.pc_encoding = pc_encoding
        self.predictor = predictor
        super().__init__()
        self.max_bounds[0] = 0.3
        self.max_bounds[1] = 0
        self.min_bounds[0] = -0.3
        self.min_bounds[1] = 0

    def cost(self, x):
        x = x[:1]
        action_param = torch.tensor(x, device=self.pc_encoding.device, dtype=torch.float32)
        pc_encoding_param = torch.concat((self.pc_encoding, action_param), dim=0)

        succ_hat = self.predictor(pc_encoding_param)

        cost = ((1 - succ_hat)**2).sum()
        return cost

    def size(self):
        return self.param_size