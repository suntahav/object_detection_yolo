""" Implementation of YOLO-v1 architecture (using batchnorm as well) """

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.batchnorm(self.cnn(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_features = in_channels
        self.darknet = self._create_cnn_layers(self.architecture)
        self.fcn = self._create_fully_connected(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcn(torch.flatten(x, start_dim=1))



    def _create_cnn_layers(self, architecture):
        layers = []
        in_channel = self.in_features

        for level in architecture:
            if type(level) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels=in_channel,
                        out_channels=level[1],
                        kernel_size=level[0],
                        stride=level[2],
                        padding=level[3]
                    )
                )
                in_channel = level[1]

            elif type(level) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

            elif type(level) == list:
                for i in range(level[-1]):
                    for j in range(len(level)-1):
                        layers.append(
                            CNNBlock(
                                in_channels=in_channel,
                                out_channels=level[j][1],
                                kernel_size=level[j][0],
                                stride=level[j][2],
                                padding=level[j][3]
                            )
                        )
                        in_channel = level[j][1]

        return nn.Sequential(*layers)

    def _create_fully_connected(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )



