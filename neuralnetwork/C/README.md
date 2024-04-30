# UQ comparison using DNN potential for carbon


## Potential description

* Atomic descriptor: Symmetry function, set 51
* Architecture:
	* Input layer with 51 nodes
	* 3 linear hidden layers with tanh activation function and dropout ratio 0.1, each having 128 nodes.
* Training: For details of the training process, see [training_dropout](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/training_dropout) folder.


## Content

The first set of files and folders are related to the setup of the calculation.
This set includes:
* [requirements.txt](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/requirements.txt), which contains the Python requirements to run the calculations.
* [settings.json](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/settings.json), which contains the settings of the calculations.
  Currently this file only controls which dataset partition to use.
  
There are also additional folders that are important in the calculation.
* [data](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/data) contains the carbon dataset.
* [energyvslatconst](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/energyvslatconst) is a Python module to compute the energy as a function of lattice parameter.
  
Finally, the main folders contain UQ calculations using various ensemble-based methods.
The results for each method are stored in separate folders.
* [training_dropout](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/training_dropout) contains a routine to perform an initial training.
  This process also exports models visited during training (see [loss_trajectory](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/loss_trajectory)).
  Additionally, this folder also contains scripts and notebooks to do UQ using Monte Carlo dropout ensemble.
* [loss_trajectory](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/loss_trajectory) generates the ensemble from the models visited during the training process.
* [fim](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/fim) contains the scripts and notebooks to do UQ using a Monte Carlo method on the weights and biases of the last layer of the NN model.
  The weights and biases are sampled from a Gaussian distribution centered at the nominal values.
  The covariance is calculated from the inverse of the FIM, which in this case is calculated by taking the expectation value of the Hessian of the log-posterior.
* [random_init](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/random_init) ensemble is obtained by randomly initializing the weights and biases at the beginning of training.
* [bootstrap](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/bootstrap) contains routines to generate bootstrap dataset and propagate the uncertainty from the ensemble.


## How to use

1. First, open [settings.json](https://github.com/yonatank93/compare_UQ/blob/main/neuralnetwork/C/settings.json) and decide whether to use "mingjian" or "yonatan" partition.
2. Then, go to [data/<partition>_partition_data](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/data) and extract the `tar.gz` file that contains the dataset.
   Follow specific instruction inside the folder, if necessary.
3. After that, start by going into [training_dropout](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/training_dropout) and doing initial training for the model.
4. After the training is done, then we can do UQ using the methods provided here.
   A recommendation is to start by using dropout ensemble method while we are in the `trainin_dropout`.
   Then, we can proceed using the [loss trajectory](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/loss_trajectory) and [Monte Carlo FIM](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/fim) methods, given that these methods are considerably cheaper than the last two methods.
   Finally, with some time and computational resources reserved, run generate [random initialization](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/random_init) and [bootstrap](https://github.com/yonatank93/compare_UQ/tree/main/neuralnetwork/C/bootstrap) ensembles.
