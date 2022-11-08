# Defending Evasion Attacks via Adversarially Adaptive Training

This code replicates the experiments for defense baselines in the paper "Defending Evasion Attacks via Adversarially AdaptiveTraining"

To run an attack, for example FGSM attack, please choose FGSM in `generate_attack.py` and run `python generate_attack.py`. This program generates data file `{attack-name}_compas.npz` which will be used in further experiments. 

Note: This code is built using implementation from IBM Adversarial Robustness Toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox). Please follow the instructions from above reference for dependencies and documentation.
	The data file "compas_data_train_test_full.npz" can be downloaded at https://tinyurl.com/yxt5e869