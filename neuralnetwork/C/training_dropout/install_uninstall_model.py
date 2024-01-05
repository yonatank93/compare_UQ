"""This script primarily contain functions and routines to install and uninstall trained
DUNN models. Since I have many DUNN models, each with different setting, I need these
functions/routines to make my life easier.
"""

from pathlib import Path
from glob import glob
import subprocess


# Directories
WORK_DIR = Path(__file__).absolute().parent
MODEL_DIR = WORK_DIR / "models"

# Iterables: Get the paths of all models inside MODEL_DIR
iterables = sorted(glob(str(MODEL_DIR / "NeuralNetwork_Dropout_C*")))


# Define functions
def install_kim_model(model_path):
    """Install exported trained KIM model. Note that we need to pass in the path of the
    folder containing the model.
    """
    subprocess.run(["kim-api-collections-management", "install", "user", model_path])


def uninstall_kim_model(model_name):
    """Uninstall or remove KIM model. Note that we only need to pass in the name of the
    model, not the path.
    """
    subprocess.run(["kim-api-collections-management", "remove", "--force", model_name])


if __name__ == "__main__":
    # Install a bunch of models
    for ii, item in enumerate(iterables):
        # Find the version given the settings
        model_path = item
        model_name = Path(item).name
        # print(ii, model_name, model_path)
        # install_kim_model(model_path)
        uninstall_kim_model(model_name)
