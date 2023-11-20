# On Variational Autoencoders: Theory and Applications

This GitHub repository is designed to accompany my master's thesis with the title "On Variational Autoencoders: Theory and Applications". All illustrations displayed throughout the figure can be found in this repository. Moreover, there can be found some code that is not displayed in the thesis, but still might be interesting to take a look at.

## Setup

It is highly recommended to setup the code as follows:
- Install `pipx` globally, e.g. using Homebrew: ```brew install pipx```
- Use `pipx` to install poetry: ```pipx install poetry```
- When `poetry` is installed, clone this repository into an empty directory and navigate to the directory. Then setup this project through ```poetry install``` in the command line. This should automatically create a fresh virtual environment and subsequently install all dependencies into the virtual environment.

Alternatively, the dependencies can be found in the ```requirements.txt```. In oder to install them please navigate into the directory, where this repository was cloned to and type in the command line ```pip install -r requirements.txt```.

## Structure of the repository

This repository is structured as follows:
- `writing`: This directory contains the entire LaTeX-code of the thesis alongside the final PDF. Moreover, all illustrarions, which are used in the thesis can be found in this directory as well.
- `data`: This directory contains the MNIST dataset, which is automatically generated upon running the models.
- `final_code`: This directory contains all models, which are described in the entire thesis. It is the heart of the application part of the thesis. We want to describe its structure in more detail in the following section.

### Structure of the `final_code` directory

The `final_code` directory is split into three subdirectories. These subdirectories are
- `linear_AE`: This subdirectory contains the entire code concerning all linear Autoencoders.
- `convolutional_AE`: This subdirectory contains the entire code concerning all convolutional Autoencoders.
- `convolutional_VAE`: This subdirectory contains the entire code concerning all Variational Autoencoders. It is the main focus of this thesis!


The structure in each of the subdirectories is quite similar. Each of the subdirectories `linear_AE`, `convolutional_AE` and `convolutional_VAE` contain multiple models. These models have slightly different architectures, which can easily be distinguished through the name of the corresponding subdirectory, e.g. `final_code/linear_AE/linear_AE_3d_amsgrad` is a linear Autoencoder with a bottleneck dimension of 3, which was optimized using the AMSGrad optimizer.\
Moreover, the `convolutional_VAE` subdirectory is split into `new_idea` and `snd`, where `snd` denotes the standard normal distribution. Since this is notation is not obvious, we describe it explicitly at this point.\
Furthermore, each model contains an `*_isa_runner.py` file, where the corresponding model is defined. The training algorithm is defined in this file as well. One might argue that this is not the best practice to keep the entire structure in one file. However, I did so because the training of the models ran on a remote machine, which I accessed using the Secure Copy (SCP) protocol. \
Furthermore, each model contains a `main.py` file. This file loads the model and computes the most important visualisations for each model. Since these visualisations vary for each model (e.g. the latent space can only be computed for models with bottleneck dimension 2 or 3). One can simply navigate from the `final_code` directory into a specific model and then run the `main.py` file.\
Additionally, each model contains a `visualise_errors.py` and `visualise_training.py` file. As the names suggest, the former visualises the average reconstruction error, where the average is taken over the entire MNIST dataset. The latter visualises the training progress, where one can see the training loss plotted against the corresponding epoch. Moreover, to highlight the trend of the training we plotted the moving average as well.
