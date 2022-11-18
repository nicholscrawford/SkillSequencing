from data_utils import get_pcparam_success_pairs
import argparse
import pickle
import os

if __name__=='__main__':
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
    "--pull_paths",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=['/home/nichols/Desktop/SkillSequnceing/Data/Nov17/TipThenPull', '/home/nichols/Desktop/SkillSequnceing/Data/Nov17/PullFromShelf'],  # default if nothing is provided
    )

    CLI.add_argument(
    "--resample",  # name on the CLI - drop the `--` for positional/required parameters
    nargs=1,  # 0 or more values expected => creates a list
    type=bool,
    default=False,  # default if nothing is provided
    )

    # parse the command line
    args = CLI.parse_args()

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