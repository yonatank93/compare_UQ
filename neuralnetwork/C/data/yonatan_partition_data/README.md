# Carbon dataset partitioned using scikit-learn

This directory contains the entire carbon dataset used by [Wen and Tadmor (2020)](https://doi.org/10.1038/s41524-020-00390-8) to train their neural network.
This partition was obtained by using scikit-learn.



## Instruction

First, uncompress the full dataset by running the following command:
``` bash
$ tar -xzvf carbon_energies_forces.tar.gz
```

Then, run `python configs_train_test_split.py` to split the entire dataset into training and test sets.
This script will create a JSON file `configs_train_test.json` that contains the information of each set.
It will also copy the training and test configuration sets into `carbon_training_set` and `carbon_test_set`, respectively.
