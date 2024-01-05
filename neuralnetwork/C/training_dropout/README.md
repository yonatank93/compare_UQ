Workflow:

1. Run `configs_train_test_split.py` to split the entire dataset into training and test sets.
   This script will create a JSON file `configs_train_test.json` that contains the information of each set.
   It will also copy the training and test configuration sets into `carbon_training_set` and `carbon_test_set`, respectively.
2. Run `initial_training.py` to train the model and get all the saved model.
   For redundancy, the saved models are exported to `models/initial_training` and `metadata/`.
3. Run `loss_evolution.py` to see how the loss value evolve during the training process.
4. Run `evaluate_loss.py` with an extra command line argument `train` or `test` to compute the loss in each saved epoch agaist the training or the test data, respectively.
   This script will also write the best KIM model, i.e., the model epoch with the lowest cost.
5. Install the best KIM model using `kim-api-collections-management install user <modelname>`
   The format the model's name is `DUNN_best_<train/test>`.
