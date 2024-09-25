from pathlib import Path
import argparse
import os
import subprocess


# Directories
FILE_DIR = Path(__file__).absolute().parent
RES_DIR = FILE_DIR / "results"

# List the directories containing the model files
model_dirs_list = [RES_DIR / f"settings0" / f"{ii:03d}" / "models" for ii in range(100)]

# Command line argument
arg_parser = argparse.ArgumentParser("Settings of the calculations")
arg_parser.add_argument("-m", "--mode", dest="mode")
args = arg_parser.parse_args()


# Iteration
for model_dir in model_dirs_list:
    # Change to model directory
    os.chdir(model_dir)
    print(model_dir)
    if args.mode == "tar":
        # # Tar
        subprocess.run("tar -czf models.tar.gz model_epoch*.pkl", shell=True)
        # Delete
        subprocess.run("rm model_epoch*.pkl", shell=True)
    elif args.mode == "untar":
        # Untar
        subprocess.run("tar -xzqf models.tar.gz", shell=True)
    else:
        raise ValueError("Mode: tar or untar")
    # Change back to the file directory
    os.chdir(FILE_DIR)
