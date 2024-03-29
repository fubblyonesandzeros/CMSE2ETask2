import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, type="2"):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1,
                    padding=1
                ),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ),

            nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=1,
                    padding=1
                ),
                nn.Dropout2d(dropout),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()

            )
        ]

        if type == "3":
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_channels, 
                        out_channels=out_channels, 
                        kernel_size=3, 
                        stride=1,
                        padding=1
                    ),
                    nn.Dropout2d(dropout),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()

                )
            )

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class VGG(nn.Module):
    def __init__(self, in_channels, dropout, classes):
        super().__init__()

        self.layers = nn.Sequential(
            VGGBlock(in_channels=in_channels, out_channels=64, dropout=dropout, type="2"),
            VGGBlock(in_channels=64, out_channels=128, dropout=dropout, type="2"),
            VGGBlock(in_channels=128, out_channels=256, dropout=dropout, type="2"),
            VGGBlock(in_channels=256, out_channels=512, dropout=dropout, type="3"),
            VGGBlock(in_channels=512, out_channels=512, dropout=dropout, type="3"),
            VGGBlock(in_channels=512, out_channels=512, dropout=dropout, type="3"),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x