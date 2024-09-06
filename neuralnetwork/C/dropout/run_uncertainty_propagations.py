"""Run (all) uncertainty propagation calculations using all architectures that we use."""

from pathlib import Path
from glob import glob
import re
import subprocess

WORK_DIR = Path(__file__).absolute().parent
ROOT_DIR = WORK_DIR.parent
SETTINGS_DIR = ROOT_DIR / "settings"
TRAIN_DIR = ROOT_DIR / "training"

# Iterables
# List of settings file paths
files = glob(str(SETTINGS_DIR / "settings*.json"))
pattern = re.compile(r"^settings\d+\.json$")
settings_path_list = filtered_files = [f for f in files if pattern.match(Path(f).name)]

# List of uncertainty propagation scripts
scripts_list = [
    "uncertainty_accuracy_energy_forces.py",
    "uncertainty_energy_vs_latconst_diamond.py",
    "uncertainty_energy_vs_latconst_graphene.py",
    "uncertainty_energy_vs_latconst_graphite.py",
    "uncertainty_latconst_ecoh_diamond.py",
    "uncertainty_latconst_ecoh_graphene.py",
    "uncertainty_latconst_ecoh_graphite.py",
    "uncertainty_phonon_dispersion_diamond.py",
    "uncertainty_phonon_dispersion_graphene.py",
    "uncertainty_phonon_dispersion_graphite.py",
    # "uncertainty_virial_stress_graphene.py",
]

# Iteration
for settings_path in settings_path_list[:5]:
    print("Settings:", settings_path)

    # Install the model for the selected architecture
    kim_command = "kim-api-collections-management"
    TRAIN_RES_DIR = TRAIN_DIR / "results" / Path(settings_path).with_suffix("").name
    modelname = "DUNN_best_train"
    subprocess.run([kim_command, "remove", "--force", modelname])  # Uninstall
    subprocess.run([kim_command, "install", "user", TRAIN_RES_DIR / modelname])  # Install

    # Prepare command line argument
    args_str = f"--path {settings_path}"
    for script in scripts_list:
        print("Script:", script)
        if script == "uncertainty_accuracy_energy_forces.py":
            # First run using the training set
            command = f"time python {script} {args_str}"
            # Then run using the test set
            for mode in ["test", "training"]:
                args_str_spec = args_str.replace(".json", "_test.json")
                command = f"time python {script} {args_str_spec}"
        else:
            command = f"time python {script} {args_str}"
        # Run
        subprocess.run(command, shell=True)

# # If running in cluster, make sure to exit after everything is done. This is so that we
# # don't just let the cluster idle
# # Get machine name
# process = subprocess.run("hostname", stdout=subprocess.PIPE)
# hostname = process.stdout.decode("utf-8")
# # Exits if running not in login node
# if not "login" in hostname:
#     subprocess.run(f"scancel $SLURM_JOB_ID", shell=True)
