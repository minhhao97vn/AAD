# Defending Evasion Attacks via Adversarially Adaptive Training

This code replicates the experiments in the paper "Defending Evasion Attacks via Adversarially AdaptiveTraining"

## Dependencies
In this project, we use python 3.7.0 and dependencies:
 - Install python 3.7.0: https://www.python.org/downloads/release/python-370/
 - Install pip for python: https://pip.pypa.io/en/stable/installation/#
 - Install dependencies: `pip install -r pip-requirement.txt`

Note: In order to avoid conflicts between the dependencies from other projects, we highly recommend using python virtual environment or Anaconda. The details can be found here: https://docs.python.org/3/tutorial/venv.html

## Source code
### Structure
To work on our source code, you might want to modify the following files:
 - `models.py`
   Models that we used in AAD experiments.
 - `aad_cnn.py`
   Functions to train adversarially adaptive model with MNIST dataset.
 -  `aad_nn.py`
   Functions to train adversarially adaptive model with COMPAS dataset.

   To run the code `aad_cnn.py` and `aad_nn.py`, use terminal with these arguments:
 
    `--lmd` Lambda value

    `--gamma` Gamma value
    
    `--seed` random seed
    
    `--batch_size` batch size

    `--meta_batch` number of meta batchs

    `--lr` learning rate

    `--gpu_id` gpu id for pytorch to use if available

    `--epochs` number of training steps

    `--model_name` name of the model to be saved

 - `data_processor.py`
   Classes and functions to build training and test data that used in our scenarios.
 - `LR_baseline.py`, `CNN_pre_baseline.py` and `CNN_dec_baseline.py`
   LR and CNN baselines in our experiments.
 - `test_white_box.py`
   Test AAD model on white box scenario.

## Note
Due to file size limitation, we only include the source code for this project. The input data for this repo can be generated following the instructions in "Reproducibility_supplementary" \\
The full source code + data can be downloaded at downloaded at https://tinyurl.com/yxt5e869 \\
We are working on this repository to make the source code more clean and easy-to-use!!!
