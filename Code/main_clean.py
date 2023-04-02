from data_utils import get_pcparam_success_pairs, get_iopairs_dict
import argparse
import pickle
import os
import torch
import numpy as np
from action_info import action
import lightning.pytorch as pl
import datetime

from precondition_prediction_clean import LitPrecondPredictor
from dataset import get_list_dataloader

if __name__=='__main__':
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--pull_paths",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[
        os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor', '2023-feb-8', 'TipThenPull'), 
        os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor', '2023-feb-8', 'PullFromShelf')
        ],  # default if nothing is provided
    #default=['/home/nichols/Desktop/SkillSequnceing/Data/SingleExample/TipThenPull', '/home/nichols/Desktop/SkillSequnceing/Data/SingleExample/PullFromShelf'],
    )

    CLI.add_argument(
    "--test_paths",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=[
        os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor', '2022-nov-20', 'TipThenPull'), 
        os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor', '2022-nov-20', 'PullFromShelf')
             ],  # default if nothing is provided
    #default=['/home/nichols/Desktop/SkillSequnceing/Data/SingleExample/TipThenPull', '/home/nichols/Desktop/SkillSequnceing/Data/SingleExample/PullFromShelf'],
    )

    CLI.add_argument(
    "--cpt_path",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=None,  # default if nothing is provided
    )

    CLI.add_argument(
    "--resample",  # name on the CLI - drop the `--` for positional/required parameters
    nargs=1,  # 0 or more values expected => creates a list
    type=bool,
    default=False,  # default if nothing is provided
    )

    # parse the command line
    args = CLI.parse_args()

    # Load I/O pairs from folders
    iopairs_dict = get_iopairs_dict(args.pull_paths)

    # Load I/O pairs for testing from folders
    test_iopairs_dict = get_iopairs_dict(args.test_paths)

    #create the cpt_path
    if args.cpt_path is None:
        today = datetime.date.today().strftime("%Y-%b-%d")
        checkpoint_path = os.path.join(os.environ['HOME'], 'model_data', 'precondition_predictor')
    else:
        checkpoint_path = args.cpt_path

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Create actions
    tasks = list(iopairs_dict.keys())
    actions = []
    for idx, task in enumerate(tasks):
        actions.append(action(task, idx, iopairs_dict[task][0][1].size))
    
    # Get dataloader
    train_dl = get_list_dataloader([iopairs_dict[action.action_name] for action in actions], actions, 64)
    train_dl.dataset.expand()
    # use 20% of training data for validation
    #train_set_size = int(len(train_dl) * 0.8)
    #valid_set_size = len(train_dl) - train_set_size

    # split the train set into two??
    #seed = torch.Generator().manual_seed(42)
    #train_set, valid_set = data.random_split(train_dl.dataset, [train_set_size, valid_set_size], generator=seed)

    test_dl = get_list_dataloader([test_iopairs_dict[action.action_name] for action in actions], actions, 64)

    # model
    predictor = LitPrecondPredictor(actions=actions, lr = 0.001)

    # train model
    trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=1, default_root_dir=checkpoint_path, max_epochs=500)
    trainer.fit(model=predictor, train_dataloaders=train_dl, val_dataloaders=test_dl)
