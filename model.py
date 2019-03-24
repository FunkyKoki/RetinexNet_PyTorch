import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomNet(nn.Module):

    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(4, channel, kernel_size*3, padding=4)
        feature_conv = []
        for idx in range(layer_num):
            feature_conv.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size, padding=1),
                nn.ReLU()
            ))
        self.conv = nn.ModuleList(feature_conv)
        self.conv1 = nn.Conv2d(channel, 4, kernel_size, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x_max = torch.max(x, dim=3, keepdim=True)
        x = torch.cat((x, x_max[0]), dim=3)
        x = x.permute(0, 3, 1, 2)

        out = self.conv0(x)
        for idx in range(self.layer_num):
            out = self.conv[idx](out)
        out = self.conv1(out)
        out = self.sig(out)

        out = out.permute(0, 2, 3, 1)
        r_part = out[:, :, :, 0:3]
        l_part = out[:, :, :, 3:4]

        return out, r_part, l_part


class RelightNet(nn.Module):

    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1),
            nn.ReLU()
        )
        self.feature_fusion = nn.Conv2d(channel*3, channel, 1)
        self.output = nn.Conv2d(channel, 1, kernel_size, padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv1(conv1)
        conv3 = self.conv1(conv2)

        up1 = F.interpolate(conv3, scale_factor=2)
        deconv1 = self.deconv1(up1) + conv2
        up2 = F.interpolate(deconv1, scale_factor=2)
        deconv2 = self.deconv2(up2) + conv1
        up3 = F.interpolate(deconv2, scale_factor=2)
        deconv3 = self.deconv3(up3) + conv0

        deconv1_resize = F.interpolate(deconv1, scale_factor=4)
        deconv2_resize = F.interpolate(deconv2, scale_factor=2)

        out = torch.cat((deconv1_resize, deconv2_resize, deconv3), dim=1)
        out = self.feature_fusion(out)
        out = self.output(out)

        return out.permute(0, 2, 3, 1)


if __name__ == '__main__':
    net = DecomNet()
    relight_net = RelightNet()
    data_in = torch.rand(1, 600, 400, 3)
    out_sum, r_low, l_low = net(data_in)
    out_S = relight_net(out_sum)
    print(out_S.size())
