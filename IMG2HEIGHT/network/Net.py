import torch
from torch import nn
from network.denseASPP import DenseASPPBlock
from network.CBAMNet import CBAM
from network.DULR_layer import DULRBlock
from network.DeformableNet import DeformConv2D
from torchvision.transforms import Resize


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


class BasicDoubleConvBlock(nn.Module):
    """ implement conv+ReLU two times """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.BatchNorm2d(middle_channels))
        conv_relu.append(nn.ReLU())

        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class ConvBlock(nn.Module):
    """docstring for ConvBlock"""

    def __init__(self, channels, stride):
        super(ConvBlock, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[3]))
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[3], kernel_size=1, stride=stride),
            nn.BatchNorm2d(channels[3]))
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.stage(x)
        # print(x1.shape)
        x2 = self.shortcut(x)
        # print(x2.shape)
        x = x1 + x2
        output = self.relu(x)
        return output


class IdentityBlock(nn.Module):
    """docstring for IdentityBlock"""

    def __init__(self, channels, stride):
        super(IdentityBlock, self).__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=1, stride=1),
            nn.BatchNorm2d(channels[3]))
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.stage(x)

        x = x1 + x
        output = self.relu(x)
        # print("output",output.shape)
        return output


class Resnet50(nn.Module):
    """docstring for Resnet50"""

    def __init__(self, ):
        super(Resnet50, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(ConvBlock([32, 32, 32, 64], stride=2),
                                    IdentityBlock([64, 32, 32, 64], stride=1))
        self.layer3 = nn.Sequential(ConvBlock([64, 64, 64, 128], stride=2),
                                    IdentityBlock([128, 64, 64, 128], stride=1))
        self.layer4 = nn.Sequential(ConvBlock([128, 128, 128, 256], stride=2),
                                    IdentityBlock([256, 128, 128, 256], stride=1))
        self.layer5 = nn.Sequential(ConvBlock([256, 256, 256, 512], stride=2),
                                    IdentityBlock([512, 256, 256, 512], stride=1))

        self.cbam1 = DULRBlock(in_out_channels=32)
        self.cbam2 = DULRBlock(in_out_channels=64)
        self.cbam3 = DULRBlock(in_out_channels=128)
        self.cbam4 = DULRBlock(in_out_channels=256)

        self.DASPP = DenseASPPBlock(512, 128, 128)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.BasicDoubleConvBlock4 = BasicDoubleConvBlock(in_channels=512 + 256, middle_channels=256, out_channels=256)
        self.BasicDoubleConvBlock3 = BasicDoubleConvBlock(in_channels=384, middle_channels=128, out_channels=128)
        self.BasicDoubleConvBlock2 = BasicDoubleConvBlock(in_channels=192, middle_channels=64, out_channels=64)

        ##可变形卷积
        self.conv11 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.offsets = nn.Conv2d(16, 18, kernel_size=3, padding=1)
        self.DF = DeformConv2D(16, 16, kernel_size=3, padding=1)

        self.final3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        self.final2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        self.final1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.layer1(x)

        x_skip1 = x
        x_skip1 = self.cbam1(x_skip1)
        x = self.layer2(x)
        x_skip2 = x
        x_skip2 = self.cbam2(x_skip2)
        x = self.layer3(x)
        x_skip3 = x
        x_skip3 = self.cbam3(x_skip3)

        x = self.layer4(x)
        x_skip4 = x
        x_skip4 = self.cbam4(x_skip4)
        x = self.layer5(x)
        x = self.DASPP(x)

        x = self.upsample(x)
        x = torch.cat((x_skip4, x), dim=1)
        x = self.BasicDoubleConvBlock4(x)
        x4 = x

        x = self.upsample(x)
        x = torch.cat((x_skip3, x), dim=1)
        x = self.BasicDoubleConvBlock3(x)
        x3 = x
        x3 = self.final3(x3)

        x = self.upsample(x)
        x = torch.cat((x_skip2, x), dim=1)
        x = self.BasicDoubleConvBlock2(x)
        x2 = x
        x2 = self.final2(x2)

        x = self.upsample(x)
        x = torch.cat((x_skip1, x), dim=1)

        x = self.conv11(x)
        offset=self.offsets(x)
        x=self.DF(x,offset)
        x = self.final1(x)

        return x, x4, x3, x2


if __name__ == '__main__':
    torch_resize2 = Resize([256, 256])
    torch_resize3 = Resize([128, 128])
    torch_resize4 = Resize([64, 64])

    criterion = nn.MSELoss()
    label = torch.rand(4, 1, 512, 512) * 20

    a = torch.rand(4, 3, 512, 512)
    # print(a)

    # a=torch.rand(4,3,512,512)
    model = Resnet50()
    output, x4, x3, x2 = model(a)

    print(output.shape, x4.shape, x3.shape, x2.shape)
    # label2 = torch_resize2(x2)
    # label3 = torch_resize3(x3)
    # label4 = torch_resize4(x4)

    # # loss=criterion(output,label)

    # loss1=criterion(output,label)
    # loss2=criterion(x2,label2)
    # loss3=criterion(x3,label3)
    # loss4=criterion(x4,label4)
    # loss=loss1+0.75*loss2+0.5*loss3+0.25*loss4
    # print(loss)

    # # torch.save(model,"./aa.pkl")
    getModelSize(model)