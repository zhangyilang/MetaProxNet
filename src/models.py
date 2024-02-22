import torch.nn as nn
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class FourBlkCNN(MetaModule):
    def __init__(self, num_classes, in_channels=3, hidden_size=32, num_feat=800):
        super(FourBlkCNN, self).__init__()
        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            nn.Flatten()
        )

        self.classifier = MetaLinear(num_feat, num_classes)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits


def res_conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.LeakyReLU(0.1)
    )


class ResBlk(MetaModule):
    def __init__(self, in_channels, out_channels):
        super(ResBlk, self).__init__()
        self.conv = MetaSequential(
            res_conv3x3(in_channels, out_channels),
            res_conv3x3(out_channels, out_channels),
            res_conv3x3(out_channels, out_channels),
        )
        self.shortcut = MetaConv2d(in_channels, out_channels, kernel_size=1)
        self.pooling = nn.MaxPool2d(2, ceil_mode=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs, params=None):
        conv_features = self.conv(inputs, params=self.get_subdict(params, 'conv'))
        res_features = self.shortcut(inputs, params=self.get_subdict(params, 'shortcut'))
        outputs = self.dropout(self.pooling(conv_features + res_features))

        return outputs


class SimpleResNet12(MetaModule):
    def __init__(self, num_classes, in_channels=3, hidden_size=(64, 96, 128, 256), num_feat=384):
        super(SimpleResNet12, self).__init__()
        self.features = MetaSequential(
            ResBlk(in_channels, hidden_size[0]),
            ResBlk(hidden_size[0], hidden_size[1]),
            ResBlk(hidden_size[1], hidden_size[2]),
            ResBlk(hidden_size[2], hidden_size[3]),
            MetaConv2d(hidden_size[3], 2048, kernel_size=1),
            nn.AvgPool2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            MetaConv2d(2048, 384, kernel_size=1),
            nn.Flatten()
        )

        self.classifier = MetaLinear(num_feat, num_classes)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits
