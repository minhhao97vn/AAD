## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST, COMPAS
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator
import utils
import numpy as np

np.random.seed(1818)


detector_I = AEDetector("./defensive_models/COMPAS_I", p=2)
reformer = SimpleReformer("./defensive_models/COMPAS_I")

id_reformer = IdReformer()
classifier = Classifier("./models/example_classifier")

detector_dict = dict()
detector_dict["I"] = detector_I

operator = Operator(COMPAS(), classifier, detector_dict, reformer)

idx = []
_, _, Y = prepare_data(COMPAS(), idx)

data = COMPAS()
testAttack = AttackData(data.test_attack_data, Y, "multiple attacks")

evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate={"I": 0.08})

