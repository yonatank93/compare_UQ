"""Run (all) uncertainty propagation calculations using all architectures that we use."""

from pathlib import Path
import subprocess

WORK_DIR = Path(__file__).absolute().parent

# Iterables
# Number of nodes in each hidden layer to try
nnodes_try = [64, 128, 196]
# Combinations of the number of nodes
nnodes_comb = [
    [n1, n2, n3] for n1 in nnodes_try for n2 in nnodes_try for n3 in nnodes_try
]
settings_list = [
    {"partition": "mingjian", "Nlayers": 4, "Nnodes": nnodes} for nnodes in nnodes_comb
]

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

# List of different prior modes
mode_list = [
    "effective_parameters_no_prior",
    "effective_parameters_weak_prior",
    "effective_parameters_strong_prior",
    # "all_parameters_no_prior",
    # "all_parameters_weak_prior",
    # "all_parameters_strong_prior",
]


for setting in settings_list:
    print("Settings:", setting)
    for mode in mode_list:
        print("Prior mode:", mode)
        # Command line arguments
        partition = setting["partition"]
        Nnodes = setting["Nnodes"]
        suffix = "_".join([str(n) for n in Nnodes])
        args_str = f"-p {setting['partition']} -l {setting['Nlayers']} -n {' '.join([str(n) for n in setting['Nnodes']])} -m {mode}"

        # Install the model for the selected architecture
        command = f"python fim_install_uninstall_model.py {args_str}"
        subprocess.run(command, shell=True)

        for script in scripts_list:
            print("Script:", script)
            if script == "uncertainty_accuracy_energy_forces.py":
                # We need to further iterate over "test" and "training" mode
                for mode in ["test", "training"]:
                    args_str_spec = args_str + f" -m {mode}"
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
