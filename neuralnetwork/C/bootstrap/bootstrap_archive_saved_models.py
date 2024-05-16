"""I want to compress and archive the saved model files. From the training, there are tons
of saved model files written. This can be used in the future. However for now, I don't
need them yet and I have a quota limit on how many files I can have in the supercomputer.
So, I will just archive them as tar files.
"""

from pathlib import Path
import json
import os
from glob import glob
import tarfile


# Read setting file
WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
with open(ROOT_DIR / "settings.json", "r") as f:
    settings = json.load(f)
partition = settings["partition"]
suffix = "_".join([str(n) for n in settings["Nnodes"]])
RES_DIR = WORK_DIR / "results" / f"{partition}_partition_{suffix}"

for ii in range(100):
    if Path(RES_DIR / f"{ii:03d}" / "last_params.npy").exists():
        print(ii)
        models_sample_dir = RES_DIR / f"{ii:03d}" / "models"
        # Change directory to the folder that store the saved models
        os.chdir(models_sample_dir)
        # List all saved model files
        saved_model_files = glob("*.pkl")

        # Tar compress
        tar_file = Path("models.tar.gz")
        if not tar_file.exists():
            with tarfile.open("models.tar.gz", "w:gz") as tar:
                for name in saved_model_files:
                    tar.add(name)
        # Remove the pickle files
        os.system("rm *.pkl")
        os.system("ls")
        os.chdir(WORK_DIR)
    else:
        print(f"Ensemble member index {ii} doesn't exists")
