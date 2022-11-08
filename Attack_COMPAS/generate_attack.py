from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras.utils.np_utils import to_categorical

tf.compat.v1.disable_eager_execution()
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.regularizers import L1L2

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, ZooAttack, SaliencyMapMethod, DeepFool, \
    ShadowAttack, BoundaryAttack, BrendelBethgeAttack, CarliniLInfMethod, DecisionTreeAttack, ElasticNet, \
    FeatureAdversariesNumpy, GeoDA, HopSkipJump, LowProFool, NewtonFool, SimBA, SpatialTransformation, SquareAttack, \
    TargetedUniversalPerturbation, UniversalPerturbation, VirtualAdversarialMethod, Wasserstein, FrameSaliencyAttack
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier

# np.random.seed(1711)

data = np.load('data/compas_data_train_test_full.npz', allow_pickle=True)

X_train = data['X_train']
X_test = data['X_test']
X_attack = data['X_attack']
y_train = data['y_train']
y_test = data['y_test']
y_attack = data['y_attack']

print(X_attack.shape)

# map_train = {}
# idx = 0
# count = 0
# for p in X_train:
#     map_train[tuple(p)] = p
#     idx += 1
#
# map_train_full = {}
# idx = 0
# count = 0
# for p in X_train_full:
#     map_train_full[tuple(p)] = p
#     idx += 1
#
# print(len(map_train))
# print(len(map_train_full))
#
# overlap_points = {k: map_train[k] for k in map_train if k in map_train_full}
# print('Num of overlap: ', len(overlap_points))
#
# exit()

y_train_cate = to_categorical(y_train)
y_test_cate = to_categorical(y_test)
y_attack_cate = to_categorical(y_attack)

print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                input_dim=X_train.shape[1]))  # input dimension = number of features your data has
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

min_, max_ = -1, 1

classifier = KerasClassifier(model=model, clip_values=None)
classifier.fit(X_train, y_train_cate, nb_epochs=10, batch_size=32)

# num_of_attack_points = 100

# X_feasible = X_test[-num_of_attack_points:]
# y_feasible = y_test[-num_of_attack_points:]

indices = np.random.choice(np.arange(0, X_attack.shape[0]), 500, replace=False)

X_feasible = X_attack[indices, :]
y_feasible = y_attack_cate[indices, :]
print('X_feasible shape: ', X_feasible.shape)


# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(X_feasible), axis=1)
acc = np.sum(preds == np.argmax(y_feasible, axis=1)) / y_feasible.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))

# Craft adversarial samples with FGSM
epsilon = 0.1  # Maximum perturbation

# adv_crafter = FastGradientMethod(classifier, eps=epsilon)
# adv_crafter = ProjectedGradientDescent(classifier, eps=epsilon)
# adv_crafter = CarliniLInfMethod(classifier)
# adv_crafter = NewtonFool(classifier)
# adv_crafter = UniversalPerturbation(classifier, delta=0.8, norm=np.inf)
# adv_crafter = VirtualAdversarialMethod(classifier, eps=epsilon)
# adv_crafter = ZooAttack(classifier, confidence=epsilon, max_iter=100, nb_parallel=1, batch_size=1)
# adv_crafter = BoundaryAttack(classifier, targeted=False)
adv_crafter = LowProFool(classifier, importance=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), eta=0.3, eta_decay=0.05) # lambd=3, n_steps=1000, threshold=0.1
# adv_crafter = DeepFool(classifier) # , epsilon=5, max_iter=500, batch_size=1


X_adv = adv_crafter.generate(x=X_feasible, y=y_feasible)

# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(X_adv), axis=1)


# acc = np.sum(preds == np.argmax(y_feasible, axis=1)) / y_feasible.shape[0]
# print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))
#
# print("\nX test: {}".format(X_test.shape))
# print("X test adv: {}".format(X_adv.shape))

np.savez('fgsm_compas.npz'.format(epsilon), X_attack=X_adv, y_attack=y_attack[indices])
