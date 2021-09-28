import torch
import torch.nn.functional as F
from torch import nn
from resnet import Resnet


class BIG(nn.Module):
    def __init__(self, channel):
        super(BIG, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(channel // 4, channel// 4, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel// 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(channel // 4, channel// 4, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel// 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(channel // 4, channel// 4, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel// 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate4 = nn.Sequential(nn.Conv2d(channel // 4, channel// 4, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel// 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channel), nn.PReLU())


        self.channel = channel
        self.weight = nn.Softmax(dim=1)

    def forward(self, x, edge):
        x1, x2, x3, x4 = torch.split(x, self.channel // 4, dim=1)

        cm1 = self.gate1(x1)
        cm2 = self.gate2(x2)
        cm3 = self.gate3(x3)
        cm4 = self.gate4(x4)

        e1 = cm1 * torch.sigmoid(edge)
        e2 = cm2 * torch.sigmoid(edge)
        e3 = cm3 * torch.sigmoid(edge)
        e4 = cm4 * torch.sigmoid(edge)

        gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))

        weight = self.weight(torch.cat((gv1, gv2, gv3, gv4), 1))
        w1, w2, w3, w4 = torch.split(weight, 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4

        return self.conv(torch.cat((nx1, nx2, nx3, nx4), 1))


class BIPGNet(nn.Module):
    def __init__(self):
        super(BIPGNet, self).__init__()
        resnet = Resnet()
        self.layer0 = resnet.layer0
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(256 * 3, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256 * 3, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(256 * 3, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )

        # SDU
        self.pr1_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pr1_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.pe1_1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pe1_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )

        # SDU
        self.pr2_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pr2_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.pe2_1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pe2_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )

        # SDU
        self.pr3_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pr3_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.pe3_1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
        )
        self.pe3_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )

        # SDU
        self.pr4_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU()
        )
        self.pr4_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.pe4_1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 256), nn.PReLU(),
        )
        self.pe4_2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )

        self.BIG1 = BIG(256)
        self.BIG2 = BIG(256)
        self.BIG3 = BIG(256)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        pr1_1 = self.pr1_1(down4)
        pr1_2 = self.pr1_2(pr1_1)
        pe1_1 = self.pe1_1(pr1_2)
        pe1_2 = self.pe1_2(pe1_1)

        pr1_1 = F.interpolate(pr1_1, size=down3.size()[2:], mode='bilinear', align_corners=True)
        pe1_1 = F.interpolate(pe1_1, size=down3.size()[2:], mode='bilinear', align_corners=True)
        pe1_2 = F.interpolate(pe1_2, size=down3.size()[2:], mode='bilinear', align_corners=True)
        fuse1 = self.fuse1(torch.cat((down3, pr1_1, pe1_1), 1))
        nfuse1 = self.BIG1(fuse1, pe1_2)
        pr2_1 = self.pr2_1(nfuse1)
        pr2_2 = self.pr2_2(pr2_1)
        pe2_1 = self.pe2_1(pr2_2)
        pe2_2 = self.pe2_2(pe2_1)

        pr2_1 = F.interpolate(pr2_1, size=down2.size()[2:], mode='bilinear', align_corners=True)
        pe2_1 = F.interpolate(pe2_1, size=down2.size()[2:], mode='bilinear', align_corners=True)
        pe2_2 = F.interpolate(pe2_2, size=down2.size()[2:], mode='bilinear', align_corners=True)
        fuse2 = self.fuse2(torch.cat((down2, pr2_1, pe2_1), 1))
        nfuse2 = self.BIG2(fuse2, pe2_2)
        pr3_1 = self.pr3_1(nfuse2)
        pr3_2 = self.pr3_2(pr3_1)
        pe3_1 = self.pe3_1(pr3_2)
        pe3_2 = self.pe3_2(pe3_1)

        pr3_1 = F.interpolate(pr3_1, size=down1.size()[2:], mode='bilinear', align_corners=True)
        pe3_1 = F.interpolate(pe3_1, size=down1.size()[2:], mode='bilinear', align_corners=True)
        pe3_2 = F.interpolate(pe3_2, size=down1.size()[2:], mode='bilinear', align_corners=True)
        fuse3 = self.fuse3(torch.cat((down1, pr3_1, pe3_1), 1))
        nfuse3 = self.BIG3(fuse3, pe3_2)
        pr4_1 = self.pr4_1(nfuse3)
        pr4_2 = self.pr4_2(pr4_1)
        pe4_1 = self.pe4_1(pr4_2)
        pe4_2 = self.pe4_2(pe4_1)

        pr1 = F.interpolate(pr1_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pr2 = F.interpolate(pr2_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pr3 = F.interpolate(pr3_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pr4 = F.interpolate(pr4_2, size=x.size()[2:], mode='bilinear', align_corners=True)

        pe1_2 = F.interpolate(pe1_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pe2_2 = F.interpolate(pe2_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pe3_2 = F.interpolate(pe3_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        pe4_2 = F.interpolate(pe4_2, size=x.size()[2:], mode='bilinear', align_corners=True)

        return pr1, pr2, pr3, pr4, pe1_2, pe2_2, pe3_2, pe4_2
