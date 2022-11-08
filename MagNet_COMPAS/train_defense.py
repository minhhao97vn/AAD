## train_defense.py
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST, COMPAS
from defensive_models import DenoisingAutoEncoder as DAE

shape = [2000, 11]
activation = "sigmoid"
reg_strength = 1e-9
epochs = 400

data = COMPAS()

AE_I = DAE(shape, v_noise=0.1, activation=activation,
           reg_strength=reg_strength)
AE_I.train(data, "COMPAS_I", num_epochs=epochs)

