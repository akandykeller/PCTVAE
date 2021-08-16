# Predictive Coding Topographic Variational Autoencoder

## Getting Started
#### Install requirements with Anaconda:
`conda env create -f environment.yml`

#### Activate the conda environment
`conda activate pctvae`

#### Install the pctvae package
Install the pctvae package inside of your conda environment. This allows you to run experiments with the `pctvae` command. At the root of the project directory run (using your environment's pip):
`pip3 install -e .`

If you need help finding your environment's pip, try `which python`, which should point you to a directory such as `.../anaconda3/envs/pctvae/bin/` where it will be located.

#### (Optional) Setup Weights & Biases:
This repository uses Weight & Biases for experiment tracking. By deafult this is set to off. However, if you would like to use this (highly recommended!) functionality, all you have to do is set `'wandb_on': True` in the experiment config, and set your account's project and entity names in the `pctvae/utils/logging.py` file.

For more information on making a Weight & Biases account see [(creating a weights and biases account)](https://app.wandb.ai/login?signup=true) and the associated [quickstart guide](https://docs.wandb.com/quickstart).

## Reproducing Results
To rerun the models from Table 1, you can run:
- VAE: `pctvae --name 'nontvae_mnist'`
- TVAE: `pctvae --name 'tvae_Lshort_mnist'`
- PCTVAE: `pctvae --name 'tvae_causal_mnist'`

To inspect the training parameters of models from Table 1, you can look at the config dictionaries contained in the files:
- VAE: 'pctvae/experiments/nontvae_mnist.py'
- TVAE: 'pctvae/experiments/tvae_Lshort_mnist.py'
- PCTVAE: 'pctvae/experiments/tvae_causal_mnist.py'

## Basics of the framework
- All models are built using the `TVAE/PCTVAE` modules (see `pctvae/containers/tvae.py`) which requires a z-encoder, a u-encoder, a decoder, and a 'grouper'. The grouper module defines the topographic structure of the latent space through a `model` (equivalent to W in the paper), and a `padder` which defines the boundary conditions.
- All experiments can be found in `pctvae/experiments/`, and begin with the model specification, followed by the experiment config where important values such as L (`group_kernel`) and K (`n_off_diag`) can be set. 


#### Model Architecutre Options
- `'n_caps'`: *int*, Number of independnt capsules
- `'cap_dim'`: *int*, Size of each capsule
- `'n_transforms'`: *int*, Length of the total transformation sequence (denoted S in the paper)
- `'mu_init'`: *int*, Initalization value for mu parameter
- `'n_off_diag'`: *int*, determines the spatial extent of the grouping within a single timestep (denoted K in the paper), `n_off_diag=1` gives K=3, while `n_off_diag=0` gives K=1.
- `'group_kernel'`: *tuple of int*, defines the size of the kernel used by the grouper, exact definition and relationship to W varies for each experiment.

#### Training Options
- `'wandb_on'`: *bool*, if True, use weights & biases logging
- `'lr'`: *float*, learning rate
- `'momentum'`: *float*, standard momentum used in SGD
- `'max_epochs'`: *int*, total training epochs
- `'eval_epochs'`: *int*, epochs between evaluation on the test (for MNIST)
- `'batch_size'`: *int*, number of samples per batch
- `'n_is_samples'`: *int*, number of importance samples when computing the log-likelihood on MNIST.


#### Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.