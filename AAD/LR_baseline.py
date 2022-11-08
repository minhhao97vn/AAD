import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data_processor import DatasetProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from matplotlib import pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import f1_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 11
hidden_size = 100
num_classes = 11
num_epochs = 60
batch_size = 50
learning_rate = 0.0001

use_output_idx = 1
task = 'Detection'
seed = 123

torch.manual_seed(seed)
# X, y = Dataset().compass_loader(shift=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, shuffle=False)
# train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
# test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
#
# print(sum(y_test)/len(y_test))
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

train_loader, train_data = DatasetProcessor().compas_poisoned_whole_loader(batch_size=50,
                                                                           data_list=['train_clean_data',
                                                                                   # 'boundary_attack',
                                                                                   # 'low_pro_fool',
                                                                                   # 'pgd',
                                                                                   # 'carlini',
                                                                                   # 'newton_fool',
                                                                                   # 'influence_attack',
                                                                                   # 'hard_examples',
                                                                                   # 'label_flipping',
                                                                                   # 'kkt_attack'
                                                                                   ], is_train=True)

test_loader, test_data = DatasetProcessor().compas_poisoned_whole_loader(batch_size=50,
                                                                         data_list=['test_clean_data', 'fgsm',
                                                                                 'newton_fool', 'pgd',
                                                                                 'deep_fool'], is_train=False)


# model for MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        reul2 = self.relu2(hidden2)
        out = self.fc3(reul2)
        return out


###### model for adaptive approach of logistic regression
class MLP_LR(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP_LR, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        reul2 = self.relu2(hidden2)
        coefficient = self.fc3(reul2)
        coefficient = torch.mean(coefficient, dim=0)
        y_predicted = torch.matmul(x, coefficient)
        y_predicted = torch.sigmoid(y_predicted)
        return y_predicted


###### model for logistic regression
class LR(torch.nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        y_predicted = self.linear(x)
        y_predicted = torch.sigmoid(y_predicted)
        return y_predicted


# model = MLP_LR(input_size, hidden_size, num_classes).to(device)


model = LR(input_size).to(device)

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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
        labels = labels[:, use_output_idx].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, torch.reshape(labels, (-1, 1)))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            train_loss.append(loss.item())

            with torch.no_grad():
                correct = 0
                total = 0

                TP_prediction = 0
                FP_prediction = 0
                FN_prediction = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels[:, use_output_idx].to(device)
                    outputs = model(images)
                    predicted = outputs.round().reshape(1, len(outputs))
                    total += labels.size(0)
                    correct += (predicted.eq(labels)).sum().item()

                    predicted = predicted.type(torch.IntTensor).numpy().reshape(-1)
                    labels = labels.type(torch.IntTensor).numpy().reshape(-1)

                    TP_prediction += np.count_nonzero(predicted & labels)
                    FP_prediction += np.count_nonzero((predicted == 0) & (labels == 1))
                    FN_prediction += np.count_nonzero((predicted == 1) & (labels == 0))
                train_acc.append(100 * correct * 1.0 / total)
                # f1_score_prediction = TP_prediction / (TP_prediction + (FP_prediction + FN_prediction) / 2)
                # train_f1.append(f1_score_prediction)
                print('Accuracy of the training example: {} %'.format(100 * correct * 1.0 / total))
                # print('F1 score of the training example: {} %'.format(f1_score_prediction))

            total_loss = 0
            number_of_points = 0
            with torch.no_grad():
                correct = 0
                total = 0
                a = 0

                TP_prediction = 0
                FP_prediction = 0
                FN_prediction = 0

                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels[:, use_output_idx].to(device)
                    outputs = model(images)
                    predicted = outputs.round().reshape(1, len(outputs))
                    # print(predicted)
                    a += torch.sum(predicted)
                    total += labels.size(0)
                    correct += (predicted.eq(labels)).sum().item()
                    number_of_points += 1
                    loss = criterion(outputs, torch.reshape(labels, (-1, 1)))
                    total_loss += loss.item()

                    predicted = predicted.type(torch.IntTensor).numpy().reshape(-1)
                    labels = labels.type(torch.IntTensor).numpy().reshape(-1)

                    TP_prediction += np.count_nonzero(predicted & labels)
                    FP_prediction += np.count_nonzero((predicted == 0) & (labels == 1))
                    FN_prediction += np.count_nonzero((predicted == 1) & (labels == 0))

                f1_score_prediction = TP_prediction / (TP_prediction + (FP_prediction + FN_prediction) / 2)

                test_loss.append(loss / number_of_points)
                test_acc.append(100 * correct * 1.0 / total)
                test_f1.append(f1_score_prediction)
                print('Accuracy of the test example: {} %'.format(100 * correct * 1.0 / total))
                print('F1 score of the test example: {} %'.format(f1_score_prediction))

# Each divided group
test_data_set = np.load('data/AAD_COMPAS_test.npz')
test_data = TensorDataset(Tensor(test_data_set['X_all']), Tensor(test_data_set['y_all']))
group_test_data = []
for idx in range(test_data_set['X_groups'].shape[0]):
    group_x = test_data_set['X_groups'][idx, :, :]
    group_y = test_data_set['y_groups'][idx, :, :]

    group_test_data.append(TensorDataset(Tensor(group_x), Tensor(group_y)))

with torch.no_grad():
    correct_clean = 0

    images, labels = test_data[-300:700]
    images = images.to(device)
    labels_pred = labels[:, 0].to(device)
    labels_dec = labels[:, 1].to(device)
    outputs = model(images)
    predicted = outputs.round().reshape(1, len(outputs))

    if task == 'Prediction':
        correct_clean += (predicted[0].eq(labels_pred)).sum().item()
    else:
        correct_clean += (predicted[0].eq(labels_dec)).sum().item()

    print('Final attack {} - Accuracy of the test example - {}: {} %'.format(1, task,
                                                                             100 * correct_clean * 1.0 / 300))

    idx = 2

    for group in group_test_data[0:4]:
        correct_attack = 0
        images, labels = group[:]
        images = images.to(device)
        labels_pred = labels[:, 0].to(device)
        labels_dec = labels[:, 1].to(device)
        outputs = model(images)
        predicted = outputs.round().reshape(1, len(outputs))

        if task == 'Prediction':
            correct_attack += (predicted[0].eq(labels_pred)).sum().item()
        else:
            correct_attack += (predicted[0].eq(labels_dec)).sum().item()

        print('Final attack {} - Accuracy of the test example - {}: {} %'.format(idx, task,
                                                                                 100 * correct_attack * 1.0 / 100))

        idx += 1

# Each attack group
# with torch.no_grad():
#     correct_clean = 0
#
#     images, labels = test_data[0:500]
#     images = images.to(device)
#     labels_pred = labels[:, 0].to(device)
#     labels_dec = labels[:, 1].to(device)
#     outputs = model(images)
#     predicted = outputs.round().reshape(1, len(outputs))
#
#     indices = (labels_dec == 0).nonzero(as_tuple=True)[0]
#     correct_clean += (predicted[0][indices].eq(labels_dec[indices])).sum().item()
#
#     print('Final attack {} - Accuracy of the test example - Prediction: {} %'.format(i + 1,
#                                                                                      100 * correct_clean * 1.0 / 500))
#
#     for i in range(4):
#         correct_attack = 0
#         images, labels = test_data[500 + i * 50:500 + i * 50 + 50]
#         images = images.to(device)
#         labels_pred = labels[:, 0].to(device)
#         labels_dec = labels[:, 1].to(device)
#         outputs = model(images)
#         predicted = outputs.round().reshape(1, len(outputs))
#
#         indices = (labels_dec == 1).nonzero(as_tuple=True)[0]
#         print(len(indices))
#         correct_attack += (predicted[0][indices].eq(labels_dec[indices])).sum().item()
#
#         print('Final attack {} - Accuracy of the test example - Prediction: {} %'.format(i + 1,
#                                                                                          100 * correct_attack * 1.0 / 50))

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
