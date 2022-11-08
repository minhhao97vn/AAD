import torch
import torch.nn as nn
import torch.nn.functional as F


class AAD_CNN(nn.Module):
    def __init__(self, img_channels, img_W, img_H, context_channels, hidden_dim, kernel_size, num_classes,
                 decoder_out_channel):
        super(AAD_CNN, self).__init__()
        self.img_channels = img_channels
        self.context_channels = context_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.num_classes = num_classes
        self.decoder_out_channel = decoder_out_channel
        self.img_W = img_W
        self.img_H = img_H

        # H(\phi) - Encoder part
        self.h_phi = nn.Sequential(
            nn.Conv2d(self.img_channels, self.hidden_dim, kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim, track_running_stats=False),
            nn.ReLU()
        )

        # Context vector c
        self.context = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.context_channels, kernel_size=(5, 5), padding=self.padding)
        )

        # f, g, r - Decoder part for prediction, detection and regularization
        self.prediction_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.prediction_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            nn.Softmax()
        )

        self.detection_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.detection_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

        self.regularization = nn.Sequential(
            nn.Linear(self.context_channels, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # H(\phi) - Encoder part
        out_h_phi = self.h_phi(x)

        # Context vector c
        out_context = self.context(out_h_phi)

        # q_p, q_d, q_r - Decoder part for prediction, detection and regularization
        prediction_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        prediction_inp = torch.repeat_interleave(prediction_inp, repeats=len(x), dim=0)
        prediction_inp_x = torch.cat((prediction_inp, x), dim=1)
        out_prediction = self.prediction_conv(prediction_inp_x)
        out_prediction = out_prediction.squeeze(dim=-1).squeeze(dim=-1)
        out_prediction = self.prediction_fc(out_prediction)

        detection_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        detection_inp = torch.repeat_interleave(detection_inp, repeats=len(x), dim=0)
        detection_inp_x = torch.cat((detection_inp, x), dim=1)
        out_detection = self.detection_conv(detection_inp_x)
        out_detection = out_detection.squeeze(dim=-1).squeeze(dim=-1)
        out_detection = self.detection_fc(out_detection)

        out_regularization = F.adaptive_avg_pool2d(out_context, 1)
        out_regularization = out_regularization.squeeze(dim=-1).squeeze(dim=-1)
        out_regularization = self.regularization(out_regularization)

        return out_prediction, out_detection, out_regularization


class AAD_CNN_Large(nn.Module):
    def __init__(self, img_channels, img_W, img_H, context_channels, hidden_dim, kernel_size, num_classes,
                 decoder_out_channel):
        super(AAD_CNN_Large, self).__init__()
        self.img_channels = img_channels
        self.context_channels = context_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.num_classes = num_classes
        self.decoder_out_channel = decoder_out_channel
        self.img_W = img_W
        self.img_H = img_H

        # H(\phi) - Encoder part
        self.h_phi = nn.Sequential(
            nn.Conv2d(self.img_channels, self.hidden_dim[0], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[0], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[1], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.hidden_dim[2], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[2], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[2], self.hidden_dim[3], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[3], track_running_stats=False),
            nn.ReLU()
        )

        # Context vector c
        self.context = nn.Sequential(
            nn.Conv2d(self.hidden_dim[3], self.context_channels, kernel_size=(5, 5), padding=self.padding)
        )

        # f, g, r - Decoder part for prediction, detection and regularization
        self.prediction_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel[0],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[0], track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel[0], self.decoder_out_channel[1],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[1], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.decoder_out_channel[1], self.decoder_out_channel[2],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[2], track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.prediction_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel[2], 1000),
            nn.ReLU(),
            nn.Linear(1000, self.num_classes),
            nn.Softmax()
        )

        self.detection_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel[0],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[0], track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel[0], self.decoder_out_channel[1],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[1], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.decoder_out_channel[1], self.decoder_out_channel[2],
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel[2], track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.detection_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel[2], 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

        self.regularization = nn.Sequential(
            nn.Linear(self.context_channels, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # H(\phi) - Encoder part
        out_h_phi = self.h_phi(x)

        # Context vector c
        out_context = self.context(out_h_phi)

        # q_p, q_d, q_r - Decoder part for prediction, detection and regularization
        prediction_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        prediction_inp = torch.repeat_interleave(prediction_inp, repeats=len(x), dim=0)
        prediction_inp_x = torch.cat((prediction_inp, x), dim=1)
        out_prediction = self.prediction_conv(prediction_inp_x)
        out_prediction = out_prediction.squeeze(dim=-1).squeeze(dim=-1)
        out_prediction = self.prediction_fc(out_prediction)

        detection_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        detection_inp = torch.repeat_interleave(detection_inp, repeats=len(x), dim=0)
        detection_inp_x = torch.cat((detection_inp, x), dim=1)
        out_detection = self.detection_conv(detection_inp_x)
        out_detection = out_detection.squeeze(dim=-1).squeeze(dim=-1)
        out_detection = self.detection_fc(out_detection)

        out_regularization = F.adaptive_avg_pool2d(out_context, 1)
        out_regularization = out_regularization.squeeze(dim=-1).squeeze(dim=-1)
        out_regularization = self.regularization(out_regularization)

        return out_prediction, out_detection, out_regularization


class AAD_ResNet(nn.Module):
    def __init__(self, img_channels, img_W, img_H, context_channels, hidden_dim, kernel_size, num_classes,
                 decoder_out_channel):
        super(AAD_ResNet, self).__init__()
        self.img_channels = img_channels
        self.context_channels = context_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.num_classes = num_classes
        self.decoder_out_channel = decoder_out_channel
        self.img_W = img_W
        self.img_H = img_H

        # H(\phi) - Encoder part
        self.h_phi = nn.Sequential(
            nn.Conv2d(self.img_channels, self.hidden_dim[0], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[0], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[1], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.hidden_dim[2], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[2], track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[2], self.hidden_dim[3], kernel_size=(5, 5), padding=self.padding),
            nn.BatchNorm2d(self.hidden_dim[3], track_running_stats=False),
            nn.ReLU()
        )

        # Context vector c
        self.context = nn.Sequential(
            nn.Conv2d(self.hidden_dim[3], self.context_channels, kernel_size=(5, 5), padding=self.padding)
        )

        # f, g, r - Decoder part for prediction, detection and regularization
        self.prediction_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.prediction_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            nn.Softmax()
        )

        self.detection_conv = nn.Sequential(
            nn.Conv2d(self.context_channels + self.img_channels, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.decoder_out_channel, self.decoder_out_channel,
                      kernel_size=(self.kernel_size, self.kernel_size)),
            nn.BatchNorm2d(self.decoder_out_channel, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.detection_fc = nn.Sequential(
            nn.Linear(self.decoder_out_channel, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

        self.regularization = nn.Sequential(
            nn.Linear(self.context_channels, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # H(\phi) - Encoder part
        out_h_phi = self.h_phi(x)

        # Context vector c
        out_context = self.context(out_h_phi)

        # q_p, q_d, q_r - Decoder part for prediction, detection and regularization
        prediction_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        prediction_inp = torch.repeat_interleave(prediction_inp, repeats=len(x), dim=0)
        prediction_inp_x = torch.cat((prediction_inp, x), dim=1)
        out_prediction = self.prediction_conv(prediction_inp_x)
        out_prediction = out_prediction.squeeze(dim=-1).squeeze(dim=-1)
        out_prediction = self.prediction_fc(out_prediction)

        detection_inp = torch.mean(out_context, dim=0).reshape((1, self.context_channels, self.img_W, self.img_H))
        detection_inp = torch.repeat_interleave(detection_inp, repeats=len(x), dim=0)
        detection_inp_x = torch.cat((detection_inp, x), dim=1)
        out_detection = self.detection_conv(detection_inp_x)
        out_detection = out_detection.squeeze(dim=-1).squeeze(dim=-1)
        out_detection = self.detection_fc(out_detection)

        out_regularization = F.adaptive_avg_pool2d(out_context, 1)
        out_regularization = out_regularization.squeeze(dim=-1).squeeze(dim=-1)
        out_regularization = self.regularization(out_regularization)

        return out_prediction, out_detection, out_regularization
