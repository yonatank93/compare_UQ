# UQ loss trajectory ensemble

The loss trajectory ensemble is obtained by sampling the models visited during the training.
To be more specific, we sample the models starting at epoch 30,0000, every 100 epoch.
We choose this starting point noting that the training loss start to plateau before this epoch.

Then, to compare the uncertainty obtained using the loss trajectory ensemble against other UQ methods, we propagate the uncertainty to some target quantities, as described below.


## Content and workflow

1. First, run `python losstraj_install_uninstall.py install` to install the KIM model members of the emsemble.
2. Uncertainty of the configuration energy per atom for the configurations in the test set.
   `uncertainty_accuracy_energies.ipynb` and `uncertainty_accuracy_energies.py` correspond to this calculation.
   The notebook has additional commands to plot the results.
3. Uncertainty of equilibrium lattice constant and cohesive energy for a monolayer graphene at 0~K.
   Use `uncertainty_latconst_ecoh.ipynb` to do this calculation.
4. Uncertainty of energy as a function of lattice parameter for a monolayer graphene at 0~K under uniform in-plane compression and stretching.
   Use `uncertainty_energy_vs_latconst.ipynb` to do this calculation.
5. Uncertainty of phonon dispersion for a monolayer graphene at 0~K along $\Gamma - M - K - \Gamma$ band path.
   The phonon calculation in `uncertainty_phonon_dispersion.ipynb` is done using ASE.
6. Uncertainty of finite temperature virial stress of graphene at 300~K.
   The ensemble calculation is done using `uncertainty_virial_stress_runmd.py` Python script, and `uncertainty_virial_stress_submitjobs.py` is used to submit the jobs to the supercomputer.
   Then, use `uncertainty_virial_stress.ipynb` to present the results.
