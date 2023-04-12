from utils.resnet import *
from math import log
import torch
import torch.nn as nn
from utils.conv_layer import *

class DPCNNblock(nn.Module):
    def __init__(self, filter_num, kernel_size, dilation):
        super(DPCNNblock, self).__init__()
        self.conv = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation, same_padding=False)
        self.max_pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        self.padding_conv = nn.ConstantPad1d(((kernel_size-1)//2)*dilation, 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

    def forward(self, x):
        x = self.padding_pool(x)
        px = self.max_pooling(x)
        # px: [batch_size, filter_num, ((len+1)-3)//2 + 1, 1-1+1=1]
        x = self.padding_conv(px)
        # x: [batch_size, filter_num, ((len+1)-3)//2+1+2 = ((len+1)-3)//2+3, 1]
        x = self.conv(x)
        # x: [batch_size, filter_num, ((len+1)-3)//2+3-3+1=((len+1)-3)//2+1, 1-1+1=1]
        x = self.padding_conv(x)
        x = self.conv1(x)
        # x: [batch_size, filter_num, ((len+1)-3)//2+3-3+1=((len+1)-3)//2+1, 1]
        x = x + px
        # x: [batch_size, filter_num, ((len+1)-3)//2+1, 1]
        return x


class DPCNN(nn.Module):

    def __init__(self, filter_num, number_of_layers):
        super(DPCNN, self).__init__()

        self.kernel_size_list = [1+x*2 for x in range(number_of_layers)]
        self.kernel_size_list = [5, 5, 5, 5, 5, 5]
        self.dilation_list = [1, 1, 1, 1, 1, 1]
        self.conv = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0]-1)//2), 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

        self.DPCNNblocklist = nn.ModuleList(
            [DPCNNblock(filter_num, kernel_size=self.kernel_size_list[i],
                        dilation=self.dilation_list[i]) for i in range(len(self.kernel_size_list))]
        )
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x):

        x = self.padding_conv(x)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        i = 0
        while x.size()[-1] > 2:
            x = self.DPCNNblocklist[i](x)
            i += 1

        x = x.squeeze(-1).squeeze(-1)

        logits = self.classifier(x)

        return logits


class multiscale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multiscale, self).__init__()

        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True)
        )

    def forward(self, x):

        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat([x0, x1, x2, x3], dim=1)
        return x4 + x

class HDRNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        base_channel = 8
        number_of_layers = int(log(101-k+1, 2))

        self.conv0 = Conv1d(768, 128, kernel_size=(1,), stride=1)
        
        self.conv1 = Conv1d(1, 128, kernel_size=(k,), stride=1, same_padding=False)

        self.multiscale_str = multiscale(128, 32)
        self.multiscale_bert = multiscale(128, 32)
        self.dpcnn = DPCNN(64 * 4, number_of_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, bert_embedding, structure):
        x0 = bert_embedding  # (N, 768, 99)
        x1 = structure  # (N, 1, 101)
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x0 = self.multiscale_bert(x0)
        x1 = self.multiscale_str(x1)
        x = torch.cat([x0, x1], dim=1)

        return self.dpcnn(x)
