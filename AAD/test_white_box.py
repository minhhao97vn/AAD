import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import Tensor
from data_processor import MNISTProcessor
from models import AAD_CNN

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

warnings.filterwarnings("ignore")

# Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument('--lmd', default=1)
parser.add_argument('--gamma', default=0.1)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--seed', default=8)
parser.add_argument('--gpu_id', default=1)
parser.add_argument('--model_name', default='aad_cnn_model.pth')
parser.add_argument('--epsilon', default=0.1)
args = parser.parse_args()

# Hyper-parameters
batch_size = int(args.batch_size)
lmd = float(args.lmd)
gamma = float(args.gamma)
seed = int(args.seed)
model_name = args.model_name
gpu_id = int(args.gpu_id)
epsilon = float(args.epsilon)

# Device configuration
device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')


def main():
    train_loader, test_loader = MNISTProcessor().mnist_poisoned_loader(device=device, batch_size=batch_size,
                                                                       train_attack_list=['fgsm',
                                                                                          'pgd',
                                                                                          'up',
                                                                                          'bond',
                                                                                          'df',
                                                                                          'jsma',
                                                                                          'hsj',
                                                                                          'gda',
                                                                                          'fsa'],
                                                                       test_attack_list=['pgd',
                                                                                         'jsma',
                                                                                         'sqa',
                                                                                         'sta'], has_white_box=True,
                                                                       epsilon=epsilon, model_name=model_name)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    img_channels = 1
    img_W = 28
    img_H = 28

    model = AAD_CNN(img_channels=img_channels, img_W=img_W, img_H=img_H, kernel_size=5, context_channels=128,
                    hidden_dim=128, decoder_out_channel=64,
                    num_classes=10).to(device)

    model.load_state_dict(torch.load('models/' + model_name))

    test_acc_prediction = []
    test_acc_detection = []

    test_acc_group_detc = [[] for i in range(7)]

    model.eval()
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

        test_acc_prediction.append(100 * correct_prediction * 1.0 / total)
        test_acc_detection.append(100 * correct_detection * 1.0 / total)

    # Group test accuracy
    with torch.no_grad():
        correct_prediction = 0
        correct_detection = 0
        total = 0

        correct_prediction_clean = 0
        correct_detection_clean = 0
        correct_prediction_attack = [0, 0, 0, 0]
        correct_detection_attack = [0, 0, 0, 0]
        total_prediction_attack = [0, 0, 0, 0]
        total_detection_attack = [0, 0, 0, 0]
        attack_idx = 0

        for i in range(len(test_loader)):
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

                total += labels_pred.size(0)
                correct_prediction += (predicted_prediction.eq(labels_pred_class)).sum().item()
                correct_detection += (predicted_detection.eq(labels_dec_class)).sum().item()

                if i < 4:
                    correct_prediction_attack[int(attack_idx)] += (
                        predicted_prediction.eq(labels_pred_class)).sum().item()
                    correct_detection_attack[int(attack_idx)] += (
                        predicted_detection.eq(labels_dec_class)).sum().item()

                    total_prediction_attack[int(attack_idx)] += len(labels_pred_class)
                    total_detection_attack[int(attack_idx)] += len(labels_dec_class)
                else:
                    correct_prediction_clean += (predicted_prediction.eq(labels_pred_class)).sum().item()
                    correct_detection_clean += (predicted_detection.eq(labels_dec_class)).sum().item()

            attack_idx += 1

        print('Final all - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction * 1.0 / total))

        print('Final all - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection * 1.0 / total))

        print('Final test clean - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction_clean * 1.0 / 6000))
        print('Final test clean - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection_clean * 1.0 / 6000))

        print('Final attack 1 - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction_attack[0] * 1.0 / total_prediction_attack[0]))
        print('Final attack 1 - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection_attack[0] * 1.0 / total_detection_attack[0]))
        print('Final attack 2 - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction_attack[1] * 1.0 / total_prediction_attack[1]))
        print('Final attack 2 - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection_attack[1] * 1.0 / total_detection_attack[1]))
        print('Final attack 3 - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction_attack[2] * 1.0 / total_prediction_attack[2]))
        print('Final attack 3 - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection_attack[2] * 1.0 / total_detection_attack[2]))
        print('Final attack 4 - Accuracy of the test example - Prediction: {} %'.format(
            100 * correct_prediction_attack[3] * 1.0 / total_prediction_attack[3]))
        print('Final attack 4 - Accuracy of the test example - Detection: {} %'.format(
            100 * correct_detection_attack[3] * 1.0 / total_detection_attack[3]))

        print(total_prediction_attack)
        print(total_detection_attack)


if __name__ == "__main__":
    main()
