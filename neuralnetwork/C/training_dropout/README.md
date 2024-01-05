Workflow:

1. Run `training.py` to train the model and get all the saved model.
   For redundancy, the saved models are exported to `results/training/<partition>/models/`
   and `results/training/<partition>/metadata/`. Execute `training_submit_job.py` to
   submit the job to train the model to the supercomputer.
2. Run `training_loss_evolution.py` to see how the loss value evolve during the training
   process.
3. Run `training_evaluate_loss.py` with an extra command line argument `train` or `test`
   to compute the loss in each saved epoch agaist the training or the test data,
   respectively. This script will also write the best KIM model, i.e., the model epoch
   with the lowest cost, and install the KIM model.
   To run this in a supercumputer, run `training_loss_submit_job.py` with the same command
   line argument.
