

import matplotlib
from sklearn.metrics import f1_score

from setup_mnist import AADMNIST

matplotlib.use('Agg')
from scipy.stats import entropy
from numpy.linalg import norm
from matplotlib.ticker import FuncFormatter
from keras.models import Sequential, load_model
from keras.activations import softmax
from keras.layers import Lambda
import numpy as np
import pylab
import os
from utils import prepare_data
import utils
import matplotlib.pyplot as plt


class AEDetector:
    def __init__(self, path, p=1):
        """
        Error based detector.
        Marks examples for filtering decisions.

        path: Path to the autoencoder used.
        p: Distance measure to use.
        """
        self.model = load_model(path)
        self.path = path
        self.p = p

    def mark(self, X):
        diff = np.abs(X - self.model.predict(X))
        marks = np.mean(np.power(diff, self.p), axis=(1, 2, 3))
        return marks

    def print(self):
        return "AEDetector:" + self.path.split("/")[-1]


class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer.
        Reforms an example to itself.
        """
        self.path = path
        self.heal = lambda X: X

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, path):
        """
        Reformer.
        Reforms examples with autoencoder. Action of reforming is called heal.

        path: Path to the autoencoder used.
        """
        self.model = load_model(path)
        self.path = path

    def heal(self, X):
        X = self.model.predict(X)
        return np.clip(X, 0.0, 1.0)

    def print(self):
        return "SimpleReformer:" + self.path.split("/")[-1]


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


class DBDetector:
    def __init__(self, reconstructor, prober, classifier, option="jsd", T=1):
        """
        Divergence-Based Detector.

        reconstructor: One autoencoder.
        prober: Another autoencoder.
        classifier: Classifier object.
        option: Measure of distance, jsd as default.
        T: Temperature to soften the classification decision.
        """
        self.prober = prober
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.option = option
        self.T = T

    def mark(self, X):
        return self.mark_jsd(X)

    def mark_jsd(self, X):
        Xp = self.prober.heal(X)
        Xr = self.reconstructor.heal(X)
        Pp = self.classifier.classify(Xp, option="prob", T=self.T)
        Pr = self.classifier.classify(Xr, option="prob", T=self.T)

        marks = [(JSD(Pp[i], Pr[i])) for i in range(len(Pr))]
        return np.array(marks)

    def print(self):
        return "Divergence-Based Detector"


class Classifier:
    def __init__(self, classifier_path):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.

        classifier_path: Path to Keras classifier file.
        """
        self.path = classifier_path
        self.model = load_model(classifier_path)
        self.softmax = Sequential()
        self.softmax.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

    def classify(self, X, option="logit", T=1):
        if option == "logit":
            return self.model.predict(X)
        if option == "prob":
            logits = self.model.predict(X) / T
            return self.softmax.predict(logits)

    def print(self):
        return "Classifier:" + self.path.split("/")[-1]


class Operator:
    def __init__(self, data, classifier, det_dict, reformer):
        """
        Operator.
        Describes the classification problem and defense.

        data: Standard problem dataset. Including train, test, and validation.
        classifier: Target classifier.
        reformer: Reformer of defense.
        det_dict: Detector(s) of defense.
        """
        self.data = data
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer
        self.normal = self.operate(AttackData(self.data.test_data,
                                              np.argmax(self.data.test_labels, axis=1), "Normal"))

    def get_thrs(self, drop_rate):
        """
        Get filtering threshold by marking validation set.
        """
        thrs = dict()
        for name, detector in self.det_dict.items():
            num = int(len(self.data.validation_data) * drop_rate[name])
            marks = detector.mark(self.data.validation_data)
            marks = np.sort(marks)
            thrs[name] = marks[-num]
        return thrs

    def operate(self, untrusted_obj):
        """
        For untrusted input(normal or adversarial), classify original input and
        reformed input. Classifier is unaware of the source of input.

        untrusted_obj: Input data.
        """
        X = untrusted_obj.data
        Y_true = untrusted_obj.labels

        X_prime = self.reformer.heal(X)
        Y = np.argmax(self.classifier.classify(X), axis=1)
        Y_judgement = (Y == Y_true[:len(X_prime)])
        Y_prime = np.argmax(self.classifier.classify(X_prime), axis=1)
        Y_prime_judgement = (Y_prime == Y_true[:len(X_prime)])

        return np.array(list(zip(np.array(Y_judgement).flatten(), np.array(Y_prime_judgement).flatten()))), Y_prime

    def filter(self, X, thrs):
        """
        untrusted_obj: Untrusted input to test against.
        thrs: Thresholds.

        return:
        all_pass: Index of examples that passed all detectors.
        collector: Number of examples that escaped each detector.
        """
        collector = dict()
        all_pass = np.array(range(10000))
        for name, detector in self.det_dict.items():
            marks = detector.mark(X)
            idx_pass = np.argwhere(marks < thrs[name])
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)
        return all_pass, collector

    def print(self):
        components = [self.reformer, self.classifier]
        return " ".join(map(lambda obj: getattr(obj, "print")(), components))


class AttackData:
    def __init__(self, examples, labels, name=""):
        """
        Input data wrapper. May be normal or adversarial.

        examples: Path or object of input examples.
        labels: Ground truth labels.
        """
        if isinstance(examples, str):
            self.data = utils.load_obj(examples)
        else:
            self.data = examples
        self.labels = labels
        self.name = name

        # print("Attack shape: {}".format(examples.shape))

    def print(self):
        return "Attack:" + self.name


class Evaluator:
    def __init__(self, operator, untrusted_data, graph_dir="./graph"):
        """
        Evaluator.
        For strategy described by operator, conducts tests on untrusted input.
        Mainly stats and plotting code. Most methods omitted for clarity.

        operator: Operator object.
        untrusted_data: Input data to test against.
        graph_dir: Where to spit the graphs.
        """
        self.operator = operator
        self.untrusted_data = untrusted_data
        self.graph_dir = graph_dir
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):
        """
        Break down of who does what in defense. Accuracy of defense on normal
        input.

        both: Both detectors and reformer take effect
        det_only: detector(s) take effect
        ref_only: Only reformer takes effect
        none: Attack effect with no defense
        """
        normal_tups = self.operator.normal
        num_normal = len(normal_tups[0])
        filtered_normal_tups = normal_tups[0][normal_all_pass]

        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC) / num_normal
        det_only_acc = sum(1 for XC, XpC in filtered_normal_tups if XC) / num_normal
        ref_only_acc = sum([1 for _, XpC in normal_tups[0] if XpC]) / num_normal
        none_acc = sum([1 for XC, _ in normal_tups[0] if XC]) / num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups[0])
        filtered_attack_tups = attack_tups[0][attack_pass]

        both_acc = 1 - sum(1 for _, XpC in filtered_attack_tups if not XpC) / num_untrusted
        det_only_acc = 1 - sum(1 for XC, XpC in filtered_attack_tups if not XC) / num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups[0] if XpC]) / num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups[0] if XC]) / num_untrusted
        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_mixed_acc(self, attack_pass, num_samples):
        attack_tups = self.data_package
        num_untrusted = len(attack_tups[0])
        filtered_attack_tups = attack_tups[0][attack_pass]

        predicted_filtered = [1 if XpC else 0 for _, XpC in filtered_attack_tups]
        predicted = np.zeros(num_untrusted)
        f_idx = 0
        for idx in attack_pass:
            predicted[idx] = predicted_filtered[f_idx]
            f_idx += 1

        both_acc = sum(predicted) / num_untrusted
        det_only_acc = 1 - sum(1 for XC, XpC in filtered_attack_tups if not XC) / num_untrusted
        ref_only_acc = sum([1 for _, XpC in attack_tups[0] if XpC]) / num_untrusted
        none_acc = sum([1 for XC, _ in attack_tups[0] if XC]) / num_untrusted

        predicted = attack_tups[1][attack_pass]

        return both_acc, det_only_acc, ref_only_acc, none_acc, predicted

    def plot_various_confidences(self, graph_name, drop_rate,
                                 idx_file="example_idx",
                                 confs=(0.0, 10.0, 20.0, 30.0, 40.0),
                                 get_attack_data_name=lambda c: "example_carlini_" + str(c)):
        """
        Test defense performance against Carlini L2 attack of various confidences.

        graph_name: Name of graph file.
        drop_rate: How many normal examples should each detector drops?
        idx_file: Index of adversarial examples in standard test set.
        confs: A series of confidence to test against.
        get_attack_data_name: Function mapping confidence to corresponding file.
        """
        pylab.rcParams['figure.figsize'] = 6, 4
        fig = plt.figure(1, (6, 4))
        ax = fig.add_subplot(1, 1, 1)

        det_only = []
        ref_only = []
        both = []
        none = []

        data = AADMNIST()

        print("\n==========================================================")
        print("Drop Rate:", drop_rate)
        thrs = self.operator.get_thrs(drop_rate)
        all_pass, _ = self.operator.filter(self.operator.data.test_data, thrs)
        all_on_acc, _, _, _ = self.get_normal_acc(all_pass)
        print("Classification accuracy with all defense on:", all_on_acc)

        # self.load_data(AttackData(data.test_group_data, np.argmax(data.test_group_pred_labels, axis=1), "all_test"))
        #
        # all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        # all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 14000)
        # print('Num pass', len(all_pass))
        # print("Classification accuracy with all defense on - all: ", all_on_acc)
        # print("Classification f1 with all defense on - all: ",
        #       f1_score(np.argmax(data.test_group_pred_labels, axis=1), predicted, average='macro'))
        # labels = data.test_group_detc_labels
        # predicted = np.ones(14000)
        # predicted[all_pass] = 0
        # acc = np.sum(predicted == labels) / 14000
        # print("Detection accuracy with all defense on - all: ", acc)
        # print("Detection f1 with all defense on - all: ", f1_score(labels, predicted))

        all_pred_predicted = []
        all_dec_predicted = []
        all_pred_labels = []
        all_dec_labels = []

        print("\n==========================================================")
        self.load_data(
            AttackData(data.test_group_data[-6000:14000], np.argmax(data.test_group_pred_labels[-6000:14000], axis=1),
                       "all_test"))
        all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 6000)
        print('Num pass', len(all_pass))
        print("Classification accuracy with all defense on - 1,2,3: ", all_on_acc)

        print("Classification f1 with all defense on - 1,2,3: ",
              f1_score(np.argmax(data.test_group_pred_labels[-6000:14000][all_pass], axis=1), predicted, average='macro'))
        labels = data.test_group_detc_labels[-6000:14000]
        dec_predicted = np.ones(6000)
        dec_predicted[all_pass] = 0
        acc = np.sum(dec_predicted == labels) / 6000
        print("Detection accuracy with all defense on - 1,2,3: ", acc)
        print("Detection f1 with all defense on - 1,2,3: ", f1_score(labels, dec_predicted))
        all_pred_predicted.append(predicted)
        all_dec_predicted.append(dec_predicted)
        all_pred_labels.append(np.argmax(data.test_group_pred_labels[-6000:14000][all_pass], axis=1))
        all_dec_labels.append(labels)

        print("\n==========================================================")
        self.load_data(
            AttackData(data.test_group_data[0:2000], np.argmax(data.test_group_pred_labels[0:2000], axis=1),
                       "all_test"))
        all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 2000)
        print('Num pass', len(all_pass))
        print("Classification accuracy with all defense on - 4: ", all_on_acc)
        print("Classification f1 with all defense on - 4: ",
              f1_score(np.argmax(data.test_group_pred_labels[0:2000][all_pass], axis=1), predicted, average='macro'))
        labels = data.test_group_detc_labels[0:2000]
        dec_predicted = np.ones(2000)
        dec_predicted[all_pass] = 0
        acc = np.sum(dec_predicted == labels) / 2000
        print("Detection accuracy with all defense on - 4: ", acc)
        print("Detection f1 with all defense on - 4: ", f1_score(labels, dec_predicted))
        all_pred_predicted.append(predicted)
        all_dec_predicted.append(dec_predicted)
        all_pred_labels.append(np.argmax(data.test_group_pred_labels[0:2000][all_pass], axis=1))
        all_dec_labels.append(labels)

        print("\n==========================================================")
        self.load_data(
            AttackData(data.test_group_data[2000:4000], np.argmax(data.test_group_pred_labels[2000:4000], axis=1),
                       "all_test"))
        all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 2000)
        print('Num pass', len(all_pass))
        print("Classification accuracy with all defense on - 5: ", all_on_acc)
        print("Classification f1 with all defense on - 5: ",
              f1_score(np.argmax(data.test_group_pred_labels[2000:4000][all_pass], axis=1), predicted, average='macro'))
        labels = data.test_group_detc_labels[2000:4000]
        dec_predicted = np.ones(2000)
        dec_predicted[all_pass] = 0
        acc = np.sum(dec_predicted == labels) / 2000
        print("Detection accuracy with all defense on - 5: ", acc)
        print("Detection f1 with all defense on - 5: ", f1_score(labels, dec_predicted))
        all_pred_predicted.append(predicted)
        all_dec_predicted.append(dec_predicted)
        all_pred_labels.append(np.argmax(data.test_group_pred_labels[2000:4000][all_pass], axis=1))
        all_dec_labels.append(labels)

        print("\n==========================================================")
        self.load_data(
            AttackData(data.test_group_data[4000:6000], np.argmax(data.test_group_pred_labels[4000:6000], axis=1),
                       "all_test"))
        all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 2000)
        print('Num pass', len(all_pass))
        print("Classification accuracy with all defense on - 6: ", all_on_acc)
        print("Classification f1 with all defense on - 6: ",
              f1_score(np.argmax(data.test_group_pred_labels[4000:6000][all_pass], axis=1), predicted, average='macro'))
        labels = data.test_group_detc_labels[4000:6000]
        dec_predicted = np.ones(2000)
        dec_predicted[all_pass] = 0
        acc = np.sum(dec_predicted == labels) / 2000
        print("Detection accuracy with all defense on - 6: ", acc)
        print("Detection f1 with all defense on - 6: ", f1_score(labels, dec_predicted))
        all_pred_predicted.append(predicted)
        all_dec_predicted.append(dec_predicted)
        all_pred_labels.append(np.argmax(data.test_group_pred_labels[4000:6000][all_pass], axis=1))
        all_dec_labels.append(labels)

        print("\n==========================================================")
        self.load_data(
            AttackData(data.test_group_data[6000:8000], np.argmax(data.test_group_pred_labels[6000:8000], axis=1),
                       "all_test"))
        all_pass, _ = self.operator.filter(self.untrusted_data.data, thrs)
        all_on_acc, _, _, _, predicted = self.get_mixed_acc(all_pass, 2000)
        print('Num pass', len(all_pass))
        print("Classification accuracy with all defense on - 7: ", all_on_acc)
        print("Classification f1 with all defense on - 7: ",
              f1_score(np.argmax(data.test_group_pred_labels[6000:8000][all_pass], axis=1), predicted, average='macro'))
        labels = data.test_group_detc_labels[6000:8000]
        dec_predicted = np.ones(2000)
        dec_predicted[all_pass] = 0
        acc = np.sum(dec_predicted == labels) / 2000
        print("Detection accuracy with all defense on - 7: ", acc)
        print("Detection f1 with all defense on - 7: ", f1_score(labels, dec_predicted))
        all_pred_predicted.append(predicted)
        all_dec_predicted.append(dec_predicted)
        all_pred_labels.append(np.argmax(data.test_group_pred_labels[6000:8000][all_pass], axis=1))
        all_dec_labels.append(labels)

        print("\n==========================================================")

        pred_predicted = np.concatenate(all_pred_predicted, axis=0)
        dec_predicted = np.concatenate(all_dec_predicted, axis=0)
        pred_labels = np.concatenate(all_pred_labels, axis=0)
        dec_labels = np.concatenate(all_dec_labels, axis=0)

        print(pred_predicted.shape)
        print(dec_predicted.shape)
        print(pred_labels.shape)
        print(dec_labels.shape)

        acc = np.sum(pred_predicted == pred_labels) / 14000
        print("Classification accuracy with all defense on - all: ", acc)
        print("Classification f1 with all defense on - 7: ",
              f1_score(pred_predicted, pred_labels, average='macro'))

        acc = np.sum(dec_predicted == dec_labels) / 14000
        print("Detection accuracy with all defense on - all: ", acc)
        print("Detection f1 with all defense on - 7: ",
              f1_score(dec_predicted, dec_labels))

    def print(self):
        return " ".join([self.operator.print(), self.untrusted_data.print()])
