

from setup_mnist import MNIST, AADMNIST
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator
import utils

seed = 1881

detector_I = AEDetector("./defensive_models/MNIST_I_seed_{}".format(seed), p=2)
detector_II = AEDetector("./defensive_models/MNIST_II_seed_{}".format(seed), p=1)
reformer = SimpleReformer("./defensive_models/MNIST_I_seed_{}".format(seed))

id_reformer = IdReformer()
classifier = Classifier("./models/example_classifier")

detector_dict = dict()
detector_dict["I"] = detector_I
detector_dict["II"] = detector_II

operator = Operator(AADMNIST(), classifier, detector_dict, reformer)

idx = []
_, _, Y = prepare_data(AADMNIST(), idx)

data = AADMNIST()

testAttack = AttackData(data.test_attack_data, Y, "multiple attacks")

evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate={"I": 0.001, "II": 0.001})

