

from setup_mnist import MNIST, AADMNIST
from defensive_models import DenoisingAutoEncoder as DAE
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

poolings = ["average", "max"]

shape = [28, 28, 1]
combination_I = [3, "average", 3]
combination_II = [3]
activation = "sigmoid"
reg_strength = 1e-9
epochs = 50

for seed in [8, 18, 2012, 1818, 1881]:
    data = AADMNIST()

    AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation,
               reg_strength=reg_strength)
    AE_I.train(data, "MNIST_I_seed_{}".format(seed), num_epochs=epochs, seed=seed)

    AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation,
                reg_strength=reg_strength)
    AE_II.train(data, "MNIST_II_seed_{}".format(seed), num_epochs=epochs, seed=seed)
