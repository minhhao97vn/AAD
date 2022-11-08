# Defending Evasion Attacks via Adversarially Adaptive Training
Submission to Big Data 2022

This code replicates the experiments for defense baselines in the paper "Defending Evasion Attacks via Adversarially AdaptiveTraining"

Note: This code is built using implementation from IBM Adversarial Robustness Toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox). Please follow the instructions from above reference for dependencies and documentation.

To run an attack, for example FGSM attack, please choose FGSM in `generate_attack.py` and run `python generate_attack.py`. This program generates data file `mnist_all.npz` and `{attack-name}_mnist.npz` which will be used in further experiments. 
