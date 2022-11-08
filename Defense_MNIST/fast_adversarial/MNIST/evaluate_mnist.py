import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

from mnist_net import mnist_net

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    y = y.to(torch.int64)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            y = y.to(torch.int64)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0 - X, 1 - X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_data = np.load('../../data/AAD_MNIST_Comb_1_test.npz')
    x_test = (test_data['X_all'] * 255).astype(np.int64)
    y_test = test_data['y_all']

    group_test_data = []

    print(test_data['y_groups'][-3:7, :, 0:-1].shape)

    x = (test_data['X_groups'][-3:7].reshape((-1, 1, 28, 28)) * 255).astype(np.int64)
    y = np.argmax(test_data['y_groups'][-3:7, :, 0:-1].reshape((-1, 10)), axis=1)

    group_test_data.append(TensorDataset(Tensor(x), Tensor(y)))

    for idx in range(4):
        group_x = (test_data['X_groups'][idx] * 255).astype(np.int64)
        group_y = test_data['y_groups'][idx, :, 0:-1]

        group_test_data.append(TensorDataset(Tensor(group_x), Tensor(np.argmax(group_y, axis=1))))

    model = mnist_net().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()

    mnist_test = TensorDataset(Tensor(x_test),
                               Tensor(np.argmax(y_test[:, 0:-1], axis=1)))
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    total_loss = 0
    total_acc = 0
    predicted = []
    labels = []
    n = 0

    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                y = y.to(torch.int64)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                predicted.append(torch.argmax(output, dim=1).cpu().numpy())
                labels.append(y.cpu().numpy())
                n += y.size(0)
    else:
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts)
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, args.epsilon)
            with torch.no_grad():
                output = model(X + delta)
                y = y.to(torch.int64)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    logger.info('Test - All - Loss: %.4f, Acc: %.4f', total_loss / n, total_acc / n)
    predicted = np.concatenate(predicted, axis=0)
    labels = np.concatenate(labels, axis=0)
    logger.info('Test - All - F1 score: %.4f', f1_score(predicted, labels, average='macro'))

    idx = 0
    for group in group_test_data:
        test_loader = torch.utils.data.DataLoader(group, batch_size=args.batch_size, shuffle=False)

        total_loss = 0
        total_acc = 0
        n = 0

        if args.attack == 'none':
            with torch.no_grad():
                for i, (X, y) in enumerate(test_loader):
                    X, y = X.cuda(), y.cuda()
                    output = model(X)
                    y = y.to(torch.int64)
                    loss = F.cross_entropy(output, y)
                    total_loss += loss.item() * y.size(0)
                    total_acc += (output.max(1)[1] == y).sum().item()
                    n += y.size(0)
        else:
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                if args.attack == 'pgd':
                    delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts)
                elif args.attack == 'fgsm':
                    delta = attack_fgsm(model, X, y, args.epsilon)
                with torch.no_grad():
                    output = model(X + delta)
                    y = y.to(torch.int64)
                    loss = F.cross_entropy(output, y)
                    total_loss += loss.item() * y.size(0)
                    total_acc += (output.max(1)[1] == y).sum().item()
                    n += y.size(0)

        logger.info('Test - {} - Loss: %.4f, Acc: %.4f'.format(idx), total_loss / n, total_acc / n)
        idx += 1


if __name__ == "__main__":
    main()
