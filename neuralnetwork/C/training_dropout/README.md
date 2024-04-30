# Initial training and UQ dropout ensemble

The scripts and notebooks here are used to do initial training of the NN potential.
At the same time, this process also exports models visited during the training that will be used in the loss trajectory method.
Additionally, we also obtain a dropout ensemble, which is streamed line through the development of this method in KLIFF.


## Potential description

* Atomic descriptor: Symmetry function, set 51
* Architecture:
	* Input layer with 51 nodes
	* 3 linear hidden layers with tanh activation function and dropout ratio 0.1, each having 128 nodes.
* Dataset:
	* Containing various carbon structures, as described in [Wen and Tadmor (2020)](https://doi.org/10.1038/s41524-020-00390-8).
	* The training energy and forces are normalized by the number of atoms in the configuration.
	* Additional weight of 0.1 is added to the forces data, as described in [Wen and Tadmor (2020)](https://doi.org/10.1038/s41524-020-00390-8).

	
## Training details

* Optimizer: Adam
* Batch size: 100
* Learning rate: $10^{-3}$ for the first 5,000 epochs, then reduced to $10^{-4}$
* Number of epochs: 40,000


## Content and workflow

Training:
1. Train the NN model by running `python training.py`.
   Alternatively, we can run `python training_submit_job.py`, which will write a slurm file and submit the training job to a supercomputer.
   (Note that some lines in the slurm file might need to be modified.)
   This process exports models visited during the training in two places, for redundancy: `results/training/<partition>_partition/models/` and `results/training/<partition>_partition/metadata/`.
2. As an analysis tool, run `python training_loss_evolution.py` to see how the loss value evolve during the training.
3. Another tool is to look at the loss computed against the training and test sets.
   Run `python training_evaluate_loss.py` with an extra command line argument `train` or `test` to compute the loss in each saved epoch agaist the training or the test data, respectively.
   Note that this calculation can be very long, since the loss are calculated by taking the mean over the dropout ensemble.
   Run `python training_loss_submit_job.py` to write a slurm file and submit the job to the super computer.
   (Note that some lines might also need t obe modified.)
   Finally, after the completion of this calculation, the script will write the best KIM model, i.e., the model epoch with the lowest cost, and install the KIM model.


UQ: To compare the uncertainty obtained using the dropout ensemble against other UQ methods, we propagate the uncertainty to some target quantities.

1. The configuration energy per atom for the configurations in the test set.
   `uncertainty_accuracy_energies.ipynb` and `uncertainty_accuracy_energies.py` correspond to this calculation.
   The notebook has additional commands to plot the results.
2. Equilibrium lattice constant and cohesive energy for a monolayer graphene at 0~K.
   Use `uncertainty_latconst_ecoh.ipynb` to do this calculation.
3. Energy as a function of lattice parameter for a monolayer graphene at 0~K under uniform in-plane compression and stretching.
   Use `uncertainty_energy_vs_latconst.ipynb` to do this calculation.
4. Phonon dispersion for a monolayer graphene at 0~K along $\Gamma - M - K - \Gamma$ band path.
   The phonon calculation in `uncertainty_phonon_dispersion.ipynb` is done using ASE.
5. Finite temperature virial stress of graphene at 300~K.
   The ensemble calculation is done using `uncertainty_virial_stress_runmd.py` Python script, and `uncertainty_virial_stress_submitjobs.py` is used to submit the jobs to the supercomputer.
   Then, use `uncertainty_virial_stress.ipynb` to present the results.
