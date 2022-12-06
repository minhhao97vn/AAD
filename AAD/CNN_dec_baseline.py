import random

import numpy as np
import torch
import torch.nn as nn
import os

from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import TensorDataset

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data_processor import MNISTProcessor

# Device configuration
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using: ", torch.cuda.get_device_name())

# Hyper-parameters
input_size = 11
hidden_size = 100
num_classes = 2
num_epochs = 1
batch_size = 64
learning_rate = 0.0002

task = 'Detection'
seed = 18

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

train_loader, train_data = MNISTProcessor().mnist_poisoned_whole_loader(batch_size=batch_size,
                                                                        attack_list=[
                                                                            'fgsm',
                                                                            'pgd',
                                                                            'up',
                                                                            'bond',
                                                                            'df',
                                                                            'jsma',
                                                                            'hsj',
                                                                            'gda',
                                                                            'fsa'
                                                                        ], is_train=True)

test_loader, test_data = MNISTProcessor().mnist_poisoned_whole_loader(batch_size=batch_size,
                                                                      attack_list=[
                                                                          'cw',
                                                                          'nf',
                                                                          'SquAttk',
                                                                          'SpaTran'
                                                                      ], is_train=False)


# model for MLP
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN, self).__init__()

        self.img_channels = 1
        self.hidden_dim = 4
        self.kernel_size = 5
        self.padding = (self.kernel_size - 1) // 2
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(self.img_channels, 4, kernel_size=(self.kernel_size, self.kernel_size)),
            # nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 10, kernel_size=(self.kernel_size, self.kernel_size)),
            # nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes)
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(dim=-1).squeeze(dim=-1)
        out = self.fc(cnn_out)
        return out


model = CNN(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
train_acc = []
test_acc = []
train_loss = []
test_loss = []
train_f1 = []
test_f1 = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels[:, -1].to(device)
        labels = torch.reshape(labels, (-1, 1))
        labels = torch.cat((1 - labels, labels), dim=1)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
    train_loss.append(loss.item())

    with torch.no_grad():
        correct = 0
        total = 0

        test_detection_labels = []
        test_detection_predicted = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels[:, -1].to(device)
            labels = torch.reshape(labels, (-1, 1))
            labels = torch.cat((1 - labels, labels), dim=1)
            outputs = model(images)

            predicted = torch.argmax(outputs, dim=1)
            labels_class = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted.eq(labels_class)).sum().item()

            test_detection_labels.append(Tensor.cpu(labels_class).numpy())
            test_detection_predicted.append(Tensor.cpu(predicted).numpy())

        train_acc.append(100 * correct * 1.0 / total)

        print('Accuracy of the training example: {} %'.format(100 * correct * 1.0 / total))
        print('F1 score of the training example: {} %'.format(
            f1_score(np.concatenate(test_detection_labels, axis=0), np.concatenate(test_detection_predicted, axis=0))))

        total_loss = 0
        number_of_points = 0

        correct = 0
        total = 0
        a = 0

        test_detection_labels = []
        test_detection_predicted = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels[:, -1].to(device)
            labels = torch.reshape(labels, (-1, 1))
            labels = torch.cat((1 - labels, labels), dim=1)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            labels_class = torch.argmax(labels, dim=1)
            a += torch.sum(predicted)
            total += labels.size(0)
            correct += (predicted.eq(labels_class)).sum().item()

            test_detection_labels.append(Tensor.cpu(labels_class).numpy())
            test_detection_predicted.append(Tensor.cpu(predicted).numpy())

        # f1_score_prediction = TP_prediction / (TP_prediction + (FP_prediction + FN_prediction) / 2)

        test_loss.append(loss / number_of_points)
        test_acc.append(100 * correct * 1.0 / total)
        # test_f1.append(f1_score_prediction)
        print('Accuracy of the test example: {} %'.format(100 * correct * 1.0 / total))
        print('F1 score of the training example: {} %'.format(
            f1_score(np.concatenate(test_detection_labels, axis=0), np.concatenate(test_detection_predicted, axis=0),
                     average='macro')))

# Each divided group
test_data_set = np.load('AAD_MNIST_Comb_1_test.npz')
test_data = TensorDataset(Tensor(test_data_set['X_all']), Tensor(test_data_set['y_all']))
group_test_data = []
for idx in range(test_data_set['X_groups'].shape[0]):
    group_x = test_data_set['X_groups'][idx, :, :]
    group_y = test_data_set['y_groups'][idx, :, :]

    group_test_data.append(TensorDataset(Tensor(group_x), Tensor(group_y)))

model.eval()
with torch.no_grad():
    correct_clean = 0

    images, labels = test_data[-6000:14000]
    images = images.to(device)
    labels = labels[:, -1].to(device)
    labels = torch.reshape(labels, (-1, 1))
    labels = torch.cat((1 - labels, labels), dim=1)
    outputs = model(images)
    predicted = torch.argmax(outputs, dim=1)
    labels_class = torch.argmax(labels, dim=1)

    correct_clean += (predicted.eq(labels_class)).sum().item()

    print('Final attack {} - Accuracy of the test example - {}: {} %'.format(1, task,
                                                                             100 * correct_clean * 1.0 / 6000))
    idx = 2

    for group in group_test_data[0:4]:
        correct_attack = 0
        images, labels = group[:]
        images = images.to(device)
        labels_pred = labels[:, 0:-1].to(device)
        labels_dec = labels[:, -1].to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        # labels_class = torch.argmax(labels_dec, dim=1)

        correct_attack += (predicted.eq(labels_dec)).sum().item()

        print('Final attack {} - Accuracy of the test example - {}: {} %'.format(idx, task,
                                                                                 100 * correct_attack * 1.0 / 2000))

        idx += 1
model.train()

# Save the model checkpoint
# plt.plot(train_acc)
# plt.show()
# plt.plot(test_acc)
# plt.show()
# plt.plot(train_loss, color='b', label='Train')
# plt.legend()
# plt.show()
# plt.plot(test_loss, color='r', label='Test')
# plt.legend()
# plt.show()
torch.save(model.state_dict(), 'model.ckpt')
