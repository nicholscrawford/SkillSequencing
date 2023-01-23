from data_utils import get_pcparam_success_pairs
import argparse
import pickle
import os
import torch
import numpy as np

#from precondition_prediction import PC_Encoder, Precond_Predictor, train
from precondition_prediction_tf import PC_Encoder, Precond_Predictor, train

if __name__=='__main__':
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--pull_paths",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=['/home/nichols/Desktop/SkillSequnceing/Data/Nov20/TipThenPull', '/home/nichols/Desktop/SkillSequnceing/Data/Nov20/PullFromShelf'],  # default if nothing is provided
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
    iopairs_dict = {}
    for path in args.pull_paths:
        iopairs_path = os.path.join(path, "io_pairs.pickle")

        if (not os.path.exists(iopairs_path)) or args.resample:
            iopairs = get_pcparam_success_pairs(path)
            with open(iopairs_path, 'wb') as fd:
                pickle.dump(iopairs, fd, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"IO Pairs saved to {path.split('/')[-1]}.")
        else:
            with open(iopairs_path, 'rb') as fd:
                iopairs = pickle.load(fd)
                print(f"IO Pairs loaded from {path.split('/')[-1]}.")
        
        iopairs_dict[path.split('/')[-1]] = iopairs

    #Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Create encoder and predictors
    enc = PC_Encoder().to(device)
    tasks = list(iopairs_dict.keys())
    list_of_predictors = {}
    for task in tasks:
        list_of_predictors[task] = Precond_Predictor(iopairs_dict[task][0][1].size).to(device)

    # Format data to tensors
    for task in tasks:
        for datum_idx in range(len(iopairs_dict[task])):
            datum = iopairs_dict[task][datum_idx]
            tensor_datum = [None for _ in datum]
            pcl = [pc for pc in list(datum[0].values())]
            np.array(pcl)
            tensor_datum[0] = torch.tensor(pcl, dtype=torch.float32, device=device)
            for i in range(1, len(datum)):
                tensor_datum[i] = torch.tensor(datum[i], device=device, dtype=torch.float32)
            iopairs_dict[task][datum_idx] = tensor_datum

    # Models saved to Checkpoints/modelname_epoch#.pth, for 1/5th of the epochs.
    train(encoder=enc, list_of_predictors=list_of_predictors, epochs=500, data=iopairs_dict, device=device)

    #TODO:
    #
    # 2. Placing books:
    #       e. write skill to place a book
    #       f. collect data for book placing
    #       g. incorperate into model
    #
    # 3. Planning:
    #       h. Use the three skills, and a desired pc, rrt to sequence actions.
    #       i. Use more data

    # pc_example = iopairs_dict[tasks[0]][0][0]
    # pcl = [pc for pc in list(pc_example.values())]
    # np.array(pcl)
    # pct = torch.tensor(pcl, dtype=torch.float32)
 
    # enc = PC_Encoder()
    # predictor = Precond_Predictor(1)
    # encoding = enc(pct)

    # #Add target and environment tags, pos 0 and 1
    # ids = torch.tensor([[1, 0], [0,0], [0, 1]])
    # encoding = torch.concat((encoding, ids), dim=1)
    
    # #Add action parameter
    # encoding = torch.reshape(encoding, [-1])
    # action_param = torch.tensor([iopairs_dict[tasks[0]][0][1]])
    # encoding = torch.concat((encoding, action_param), dim=0)

    # prediciton = predictor(encoding)
    # print(encoding)
    # print(prediciton)