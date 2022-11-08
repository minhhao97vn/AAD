import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from models import AAD_CNN

warnings.filterwarnings("ignore")


def attack_fgsm_pt(model, image, label_pred, label_dec, epsilon):
    image.requires_grad = True
    y_pred, y_dec, y_context = model(image)
    loss = F.cross_entropy(y_pred, label_pred) + 0.1 * F.cross_entropy(y_dec, label_dec) + 0.1 * torch.abs(
        F.cross_entropy(y_pred, label_pred) - F.kl_div(y_pred, y_context))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    sign_data_grad = data_grad.sign()
    pert_image = image + epsilon * sign_data_grad
    pert_image = torch.clamp(pert_image, 0, 1)

    return pert_image


def attack_fgsm(model, image, label_pred, label_dec, epsilon):
    delta = torch.zeros_like(image, requires_grad=True)
    y_pred, y_dec, y_context = model(image + delta)
    loss = F.cross_entropy(y_pred, label_pred) + 0.1 * F.cross_entropy(y_dec, label_dec) + 0.1 * torch.abs(
        F.cross_entropy(y_pred, label_pred) - F.kl_div(y_pred, y_context))
    # model.zero_grad()
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(device, model, image, label_pred, label_dec, epsilon, attack_iters, restarts, alpha):
    max_loss = torch.zeros(label_pred.shape[0]).to(device)
    max_delta = torch.zeros_like(image).to(device)
    for res in range(restarts):
        # print("Restart: {}/{}".format(res, restarts))
        delta = torch.zeros_like(image).uniform_(-epsilon, epsilon).to(device)
        delta.data = clamp(delta, 0 - image, 1 - image)
        delta.requires_grad = True
        for _ in range(attack_iters):
            y_pred, y_dec, y_context = model(image + delta)
            index = torch.where(y_pred.max(1)[1] == label_pred.max(1)[1])[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(y_pred, label_pred) + 0.1 * F.cross_entropy(y_dec, label_dec) + 0.1 * torch.abs(
                F.cross_entropy(y_pred, label_pred) - F.kl_div(y_pred, y_context))
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0 - image, 1 - image)
            delta.data[index] = d[index]
            delta.grad.zero_()
        y_pred, y_dec, y_context = model(image + delta)
        all_loss = F.cross_entropy(y_pred, label_pred, reduction='none') + \
                   0.1 * F.cross_entropy(y_dec, label_dec, reduction='none') + \
                   0.1 * torch.abs(
            F.cross_entropy(y_pred, label_pred, reduction='none') - F.kl_div(y_pred, y_context, reduction='none').mean(
                dim=1))
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    # print(max_delta[0])
    return max_delta


def generate_attack(num_samples, epsilon, device, seed, model_name):
    from data_processor import MNISTProcessor

    data_loader = MNISTProcessor().mnist_feasible_attack_loader(batch_size=64, num_samples=num_samples)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    print('Load model {} for generating attack'.format(model_name))

    img_channels = 1
    img_W = 28
    img_H = 28

    model = AAD_CNN(img_channels=img_channels, img_W=img_W, img_H=img_H, kernel_size=5, context_channels=128,
                    hidden_dim=128, decoder_out_channel=64,
                    num_classes=10).to(device)

    model.load_state_dict(torch.load('models/' + model_name))

    model.eval()

    x_adv = []
    y_adv = []

    count = 1

    for image, labels in data_loader:
        image = image.to(device)
        labels_pred = labels[:, 0:-1].to(device)
        labels_dec = labels[:, -1].to(device)
        labels_dec = torch.reshape(labels_dec, (-1, 1))
        labels_dec = torch.cat((1 - labels_dec, labels_dec), dim=1)

        # delta = attack_fgsm(model, image, labels_pred, labels_dec, epsilon)
        delta = attack_pgd(device, model, image, labels_pred, labels_dec, epsilon, 100, 1, 0.1)
        x_adv.append(torch.clamp(image + delta, 0, 1).cpu().numpy())
        y_adv.append(labels.numpy())

        # pert_image = attack_fgsm_pt(model, image, labels_pred, labels_dec, epsilon)
        # x_adv.append(pert_image.cpu().detach().numpy())
        # y_adv.append(labels.numpy())

        print('Generated {}/{}'.format(count * 64, num_samples))
        count += 1

    x_adv = np.concatenate(x_adv, axis=0)
    y_adv = np.concatenate(y_adv, axis=0)

    print(x_adv.shape)
    print(y_adv.shape)

    np.random.seed(seed)

    return x_adv, y_adv[:, 0:-1]
