import argparse
import os
import random
import warnings

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor

from data_processor import MNISTProcessor, CIFAR10Processor
from models import AAD_CNN, AAD_CNN_Large
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--lmd', default=1)
parser.add_argument('--gamma', default=0.1)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--meta_batch', default=3)
parser.add_argument('--seed', default=8)
parser.add_argument('--lr', default=0.0003)
parser.add_argument('--gpu_id', default=1)
parser.add_argument('--epochs', default=2000)
parser.add_argument('--model_name', default='aad_cnn_model.pth')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--dataset', default='MNIST')
args = parser.parse_args()

# Hyper-parameters
num_epochs = int(args.epochs)
batch_size = int(args.batch_size)
learning_rate = float(args.lr)
meta_batch = int(args.meta_batch)
lmd = float(args.lmd)
gamma = float(args.gamma)
seed = int(args.seed)
model_name = args.model_name
gpu_id = int(args.gpu_id)
dataset = args.dataset

verbose = args.verbose

# Device configuration
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
print("Using: GPU-{}".format(gpu_id), torch.cuda.get_device_name())


def main():
    if dataset == 'MNIST':
        # Use this loader for single-attack test scenario, ratio of test data is defined in loader function
        loader = MNISTProcessor().mnist_poisoned_loader

        # Use this loader for multi-attack test scenario, ratio of test data is defined in loader function
        # loader = MNISTProcessor().mnist_poisoned_group_test_multi_attacks_loader
    else:
        loader = CIFAR10Processor().cifar_poisoned_loader

    train_loader, test_loader = loader(device=device, batch_size=batch_size,
                                       train_attack_list=[
                                           'fgsm',
                                           'pgd',
                                           'up',
                                           'bond',
                                           'df',
                                           'jsma',
                                           'hsj',
                                           'gda',
                                           'fsa'],
                                       test_attack_list=['cw',
                                                         'nf',
                                                         'sqa',
                                                         'sta'],
                                       has_white_box=False)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if dataset == 'MNIST':
        img_channels = 1
        img_W = 28
        img_H = 28
    else:
        img_channels = 3
        img_W = 32
        img_H = 32

    if dataset == 'MNIST':
        model = AAD_CNN(img_channels=img_channels, img_W=img_W, img_H=img_H, kernel_size=5, context_channels=128,
                        hidden_dim=128,
                        decoder_out_channel=64,
                        num_classes=10).to(device)
    else:
        model = AAD_CNN_Large(img_channels=img_channels, img_W=img_W, img_H=img_H, kernel_size=5, context_channels=128,
                              hidden_dim=[64, 128, 128, 256],
                              decoder_out_channel=[256, 512, 512],
                              num_classes=10).to(device)

    # Loss and optimizer
    criterion_prediction = torch.nn.CrossEntropyLoss()
    criterion_detection = torch.nn.CrossEntropyLoss()
    criterion_context = torch.nn.KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    train_loss = []
    train_acc_prediction = []
    train_acc_detection = []
    test_acc_prediction = []
    test_acc_detection = []


    train_acc_group_detc = [[] for i in range(10)]
    test_acc_group_detc = [[] for i in range(7)]

    output = "Setting: lambda = {}, gamma = {}, seed = {} \n".format(lmd, gamma, seed)

    start_time = time.time()

    for epoch in range(num_epochs):
        group_ids = random.sample(range(0, len(train_loader)), meta_batch)
        loss = 0

        for index in group_ids:
            images, labels = next(iter(train_loader[index]))
            images = images.to(device)
            labels_pred = labels[:, 0:-1].to(device)
            labels_dec = labels[:, -1].to(device)

            labels_dec = torch.reshape(labels_dec, (-1, 1))
            labels_dec = torch.cat((1 - labels_dec, labels_dec), dim=1)


            # Forward pass
            y_pre, y_dec, y_context = model(images)
            loss_prediction = criterion_prediction(y_pre, torch.reshape(labels_pred, (-1, 10)))
            loss_detection = criterion_detection(y_dec, torch.reshape(labels_dec, (-1, 2)))
            loss_context = criterion_context(y_pre, y_context)
            loss += loss_prediction + lmd * loss_detection + gamma * torch.abs(loss_prediction - loss_context)

        loss = loss / meta_batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print('Epoch [{}/{}], Total Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, loss.item()))

            model.eval()
            with torch.no_grad():
                correct_prediction = 0
                correct_detection = 0
                total = 0

                for i in range(len(train_loader)):

                    group_total = 0
                    group_correct_detection = 0
                    training_prediction_labels = []
                    training_detection_labels = []
                    training_prediction_predicted = []
                    training_detection_predicted = []

                    for images, labels in train_loader[i]:
                        images = images.to(device)
                        labels_pred = labels[:, 0:-1].to(device)
                        labels_dec = labels[:, -1].to(device)
                        labels_dec = torch.reshape(labels_dec, (-1, 1))
                        labels_dec = torch.cat((1 - labels_dec, labels_dec), dim=1)

                        y_pre, y_dec, y_context = model(images)

                        predicted_prediction = torch.argmax(y_pre, dim=1)
                        predicted_detection = torch.argmax(y_dec, dim=1)
                        labels_pred_class = torch.argmax(labels_pred, dim=1)
                        labels_dec_class = torch.argmax(labels_dec, dim=1)

                        training_prediction_labels.append(Tensor.cpu(labels[:, 0:-1]).numpy())
                        training_detection_labels.append(Tensor.cpu(labels[:, -1]).numpy())
                        training_prediction_predicted.append(np.eye(10)[Tensor.cpu(predicted_prediction).numpy()])
                        training_detection_predicted.append(Tensor.cpu(predicted_detection).numpy())

                        total += labels_pred.size(0)
                        correct_prediction += (predicted_prediction.eq(labels_pred_class)).sum().item()
                        correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                        group_total += labels_pred.size(0)
                        group_correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                    train_acc_group_detc[i].append(100 * group_correct_detection * 1.0 / group_total)

                train_loss.append(loss.item())

                print('Accuracy of the training example - Prediction: {} %'.format(
                    100 * correct_prediction * 1.0 / total))
                print('F1 score of the training example - Prediction: {}'.format(
                    f1_score(np.concatenate(training_prediction_labels, axis=0),
                             np.concatenate(training_prediction_predicted, axis=0), average='macro')))
                print(
                    'Accuracy of the training example - Detection: {} %'.format(100 * correct_detection * 1.0 / total))
                print('F1 score of the training example - Detection: {}'.format(
                    f1_score(np.concatenate(training_detection_labels, axis=0),
                             np.concatenate(training_detection_predicted, axis=0), average='binary')))

                output += 'Accuracy of the training example - Prediction: {} % \n'.format(
                    100 * correct_prediction * 1.0 / total)

                output += 'Accuracy of the training example - Detection: {} % \n'.format(
                    100 * correct_detection * 1.0 / total)

                train_acc_prediction.append(100 * correct_prediction * 1.0 / total)
                train_acc_detection.append(100 * correct_detection * 1.0 / total)
            #
            with torch.no_grad():
                correct_prediction = 0
                correct_detection = 0
                total = 0

                test_prediction_labels = []
                test_detection_labels = []
                test_prediction_predicted = []
                test_detection_predicted = []

                for i in range(len(test_loader)):
                    group_correct_detection = 0
                    group_total = 0
                    for images, labels in test_loader[i]:
                        images = images.to(device)
                        labels_pred = labels[:, 0:-1].to(device)
                        labels_dec = labels[:, -1].to(device)
                        labels_dec = torch.reshape(labels_dec, (-1, 1))
                        labels_dec = torch.cat((1 - labels_dec, labels_dec), dim=1)

                        test_prediction_labels.append(Tensor.cpu(labels[:, 0:-1]).numpy())
                        test_detection_labels.append(Tensor.cpu(labels[:, -1]).numpy())

                        y_pre, y_dec, y_context = model(images)

                        predicted_prediction = torch.argmax(y_pre, dim=1)
                        predicted_detection = torch.argmax(y_dec, dim=1)
                        labels_pred_class = torch.argmax(labels_pred, dim=1)
                        labels_dec_class = torch.argmax(labels_dec, dim=1)

                        test_prediction_predicted.append(np.eye(10)[Tensor.cpu(predicted_prediction).numpy()])
                        test_detection_predicted.append(Tensor.cpu(predicted_detection).numpy())

                        group_total += labels_pred.size(0)
                        # group_correct_prediction += (predicted_prediction.eq(labels_pred)).sum().item()
                        group_correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                        total += labels_pred.size(0)
                        correct_prediction += (predicted_prediction.eq(labels_pred_class)).sum().item()
                        correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                    test_acc_group_detc[i].append(100 * group_correct_detection * 1.0 / group_total)
                    # if True:
                print('Accuracy of the test example - Prediction: {} %'.format(100 * correct_prediction * 1.0 / total))
                print('F1 score of the test example - Prediction: {}'.format(
                    f1_score(np.concatenate(test_prediction_labels, axis=0),
                             np.concatenate(test_prediction_predicted, axis=0), average='macro')))
                print('Accuracy of the test example - Detection: {} %'.format(100 * correct_detection * 1.0 / total))
                print('F1 score of the test example - Detection: {}'.format(
                    f1_score(np.concatenate(test_detection_labels, axis=0),
                             np.concatenate(test_detection_predicted, axis=0), average='binary')))

                output += 'Accuracy of the test example - Prediction: {} % \n'.format(
                    100 * correct_prediction * 1.0 / total)

                output += 'Accuracy of the test example - Detection: {} % \n'.format(
                    100 * correct_detection * 1.0 / total)

                test_acc_prediction.append(100 * correct_prediction * 1.0 / total)
                test_acc_detection.append(100 * correct_detection * 1.0 / total)

            model.train()

    end_time = time.time()

    print("Exec. Time: {}".format(end_time - start_time))

    # Group test accuracy
    model.eval()
    with torch.no_grad():
        correct_prediction = 0
        correct_detection = 0
        total = 0

        correct_prediction_attack = [0, 0, 0, 0, 0, 0, 0]
        correct_detection_attack = [0, 0, 0, 0, 0, 0, 0]
        attack_idx = 0

        test_prediction_labels = []
        test_detection_labels = []
        test_prediction_predicted = []
        test_detection_predicted = []

        for i in range(len(test_loader)):
            total_group = 0

            for images, labels in test_loader[i]:
                images = images.to(device)
                labels_pred = labels[:, 0:-1].to(device)
                labels_dec = labels[:, -1].to(device)
                labels_dec = torch.reshape(labels_dec, (-1, 1))
                labels_dec = torch.cat((1 - labels_dec, labels_dec), dim=1)

                test_prediction_labels.append(Tensor.cpu(labels[:, 0:-1]).numpy())
                test_detection_labels.append(Tensor.cpu(labels[:, -1]).numpy())

                y_pre, y_dec, y_context = model(images)

                predicted_prediction = torch.argmax(y_pre, dim=1)
                predicted_detection = torch.argmax(y_dec, dim=1)
                labels_pred_class = torch.argmax(labels_pred, dim=1)
                labels_dec_class = torch.argmax(labels_dec, dim=1)

                test_prediction_predicted.append(np.eye(10)[Tensor.cpu(predicted_prediction).numpy()])
                test_detection_predicted.append(Tensor.cpu(predicted_detection).numpy())

                total += labels_pred.size(0)
                correct_prediction += (predicted_prediction.eq(labels_pred_class)).sum().item()
                correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                correct_prediction_attack[int(attack_idx)] += (
                    predicted_prediction.eq(labels_pred_class)).sum().item()
                correct_detection_attack[int(attack_idx)] += (
                    predicted_detection.eq(labels_dec_class)).sum().item()

                total_group += labels_pred.size(0)

            print('Group {} - Accuracy of the test example - Prediction: {} %'.format(i,
                                                                                      100 * correct_prediction_attack[
                                                                                          i] * 1.0 / total_group))
            print('Group {} - Accuracy of the test example - Detection: {} %'.format(i, 100 * correct_detection_attack[
                i] * 1.0 / total_group))

            attack_idx += 1

        print('Final all - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction * 1.0 / total))
        print('Final all - F1 score of the test example - Prediction: {} %'.format(
            f1_score(np.concatenate(test_prediction_labels, axis=0),
                     np.concatenate(test_prediction_predicted, axis=0), average='macro')))

        print('Final all - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection * 1.0 / total))
        print('Final all - F1 score of the test example - Detection: {} %'.format(
            f1_score(np.concatenate(test_detection_labels, axis=0),
                     np.concatenate(test_detection_predicted, axis=0), average='binary')))

    model.train()

    # Save the model checkpoint
    torch.save(model.state_dict(), 'models/' + model_name)


if __name__ == "__main__":
    main()
