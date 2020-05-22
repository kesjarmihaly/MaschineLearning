import torch
from torch import nn
from torchvision.models.vgg import vgg19

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, channels, upscale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(64)
        self.res_block4 = ResidualBlock(64)
        self.res_block5 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

        self.up1 = UpsampleBlock(64, upscale_factor//2)
        self.up2 = UpsampleBlock(64, upscale_factor//2)
        self.conv3d = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        identity = self.prelu(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += identity
        out = self.up1(out)
        out = self.up2(out)
        out = self.conv3d(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, patch_size=96):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.block1 = self.conv_bn_lrelu( 64,  64, 2)
        self.block2 = self.conv_bn_lrelu( 64, 128, 1)
        self.block3 = self.conv_bn_lrelu(128, 128, 2)
        self.block4 = self.conv_bn_lrelu(128, 256, 1)
        self.block5 = self.conv_bn_lrelu(256, 256, 2)
        self.block6 = self.conv_bn_lrelu(256, 512, 1)
        self.block7 = self.conv_bn_lrelu(512, 512, 2)
        self.flat = Flatten()
        self.dense1 = nn.Linear(512 * (patch_size//2 ** 4) ** 2, 1024)
        self.dense2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def conv_bn_lrelu(self, in_ch, out_ch, stride):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.flat(out)
        out = self.dense1(out)
        out = self.lrelu(out)
        out = self.dense2(out)
        out = self.sigmoid(out)
        return out


class GeneratorLoss(nn.Module):
    def __init__(self, loss_type='vgg22', adv_coefficient=1e-3):
        super(GeneratorLoss, self).__init__()
        self.content_loss = VGGLoss(loss_type)
        self.mse_loss = nn.MSELoss()
        self.adv_coefficient = adv_coefficient

    def forward(self, D_out_fake, real_img, fake_img):
        mse_loss = self.mse_loss(real_img, fake_img)
        content_loss = self.content_loss(real_img, fake_img)
        adv_loss = torch.mean(-torch.log(D_out_fake + 1e-3))
        return mse_loss + 2e-6 * content_loss + self.adv_coefficient * adv_loss


class VGGLoss(nn.Module):
    def __init__(self, loss_type):
        super(VGGLoss, self).__init__()
        vgg = vgg19(pretrained=True)


        if loss_type == 'vgg22':
            vgg_net = nn.Sequential(*list(vgg.features[:9]))
        elif loss_type == 'vgg54':
            vgg_net = nn.Sequential(*list(vgg.features[:36]))

        for param in vgg_net.parameters():
            param.requires_grad = False

        self.vgg_net = vgg_net.eval()
        self.mse_loss = nn.MSELoss()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406], requires_grad=False))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225], requires_grad=False))


    def forward(self, real_img, fake_img):
        # バッチ単位で正規化, 平均引いて標準偏差で割る.
        real_img = real_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])
        fake_img = fake_img.sub(self.vgg_mean[:, None, None]).div(self.vgg_std[:, None, None])

        feature_real = self.vgg_net(real_img)
        feature_fake = self.vgg_net(fake_img)
        return self.mse_loss(feature_real, feature_fake)


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, D_out_real, D_out_fake):
        loss_real = self.bce_loss(D_out_real, torch.ones_like(D_out_real))
        loss_fake = self.bce_loss(D_out_fake, torch.zeros_like(D_out_fake))
        return loss_real + loss_fake