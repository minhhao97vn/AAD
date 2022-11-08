"""
The code is revised from IBM ART
"""
import tensorflow as tf

from split_train_attack_test import load_mnist_train_attack_test

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, TargetedUniversalPerturbation, ElasticNet, \
    CarliniLInfMethod, NewtonFool, UniversalPerturbation, VirtualAdversarialMethod, ZooAttack, CarliniL2Method, \
    BoundaryAttack, LowProFool, DeepFool, Wasserstein, SquareAttack, SpatialTransformation, SaliencyMapMethod, \
    HopSkipJump, GeoDA, FrameSaliencyAttack, FeatureAdversariesNumpy, CarliniL0Method
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist

(x_train, y_train), (x_test, y_test), (
    x_attack, y_attack), min_pixel_value, max_pixel_value = load_mnist_train_attack_test()

print("X_train shape: ", x_train.shape)
print("X_test shape:", x_test.shape)
print("X_attack shape: ", x_attack.shape)

# Step 2: Create the model

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

attack = FastGradientMethod(estimator=classifier)
# attack = ProjectedGradientDescent(estimator=classifier) # take time
# attack = CarliniL2Method(classifier)
# attack = NewtonFool(classifier)
# attack = UniversalPerturbation(classifier)
# attack = BoundaryAttack(classifier, targeted=False)
# attack = DeepFool(classifier)
# attack = SquareAttack(classifier, nb_restarts=50)
# attack = SpatialTransformation(classifier, max_translation=10, num_translations=3, max_rotation=18, num_rotations=3)
# attack = SaliencyMapMethod(classifier, gamma=0.15)
# attack = HopSkipJump(classifier)
# attack = GeoDA(classifier)
# attack = FrameSaliencyAttack(classifier, FastGradientMethod(estimator=classifier))

indices = np.random.choice(np.arange(0, x_attack.shape[0]), 5000, replace=False)
x_feasible = x_attack[indices, :, :, :]
y_feasible = y_attack[indices, :]

# Provide y=y_feasible if attack is white-box
x_adv = attack.generate(x=x_feasible, y=y_feasible)

# Enable this code to print out attack performance
# print("X_adv shape: ", x_adv.shape)

# predictions = classifier.predict(x_feasible)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_feasible, axis=1)) / len(y_feasible)
# print("Accuracy on clean test examples: {}%".format(accuracy * 100))
#
# predictions = classifier.predict(x_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_feasible, axis=1)) / len(y_feasible)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
np.savez('MNIST_adv/fgsm_mnist.npz', X_attack=x_adv, y_attack=y_feasible)

print("\n\n\n\n")
