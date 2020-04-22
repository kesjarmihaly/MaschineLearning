import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import load_datasets
from pix2pix import Generator, Discriminator

import statistics
import os
from tqdm import tqdm

def train():
    # モデル
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter()

    model_G, model_D = Generator(), Discriminator()
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G, model_D = model_G.to(device), model_D.to(device)

    params_G = torch.optim.Adam(model_G.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(model_D.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))

    # ロスを計算するためのラベル変数 (PatchGAN)
    ones = torch.ones(512, 1, 3, 3).to(device)
    zeros = torch.zeros(512, 1, 3, 3).to(device)

    # 損失関数
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()

    # エラー推移
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []

    # 訓練
    dataset = load_datasets()

    for i in range(50):
        log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []

        for (real_color, input_gray), _ in tqdm(dataset):

            batch_len = len(real_color)
            real_color, input_gray = real_color.to(device), input_gray.to(device)

            # Gの訓練
            # 偽のカラー画像を作成
            fake_color = model_G(input_gray)

            # 偽画像を一時保存
            fake_color_tensor = fake_color.detach()

            # 偽画像を本物と騙せるようにロスを計算
            LAMBD = 100.0  # BCEとMAEの係数
            out = model_D(torch.cat([fake_color, input_gray], dim=1))
            loss_G_bce = bce_loss(out, ones[:batch_len])
            loss_G_mae = LAMBD * mae_loss(fake_color, real_color)
            loss_G_sum = loss_G_bce + loss_G_mae

            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_G_sum.backward()
            params_G.step()

            # Discriminatoの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = model_D(torch.cat([real_color, input_gray], dim=1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = model_D(torch.cat([fake_color_tensor, input_gray], dim=1))
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])


            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            params_D.zero_grad()
            params_G.zero_grad()
            loss_D.backward()
            params_D.step()

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))

        writer.add_scalar('Loss/log_loss_G_sum', result['log_loss_G_sum'][-1], i)
        writer.add_scalar('Loss/log_loss_G_bce', result['log_loss_G_bce'][-1], i)
        writer.add_scalar('Loss/log_loss_G_mae', result['log_loss_G_mae'][-1], i)
        writer.add_scalar('Loss/log_loss_D', result['log_loss_D'][-1], i)

        # 画像を保存
        if not os.path.exists("stl_color"):
            os.mkdir("stl_color")
        # 生成画像を保存
        torchvision.utils.save_image(fake_color_tensor[:min(batch_len, 100)],
                                     f"stl_color/fake_epoch_{i:03}.png",
                                     range=(-1.0, 1.0), normalize=True)
        torchvision.utils.save_image(real_color[:min(batch_len, 100)],
                                     f"stl_color/real_epoch_{i:03}.png",
                                     range=(-1.0, 1.0), normalize=True)

        # モデルの保存
        if not os.path.exists("stl_color/models"):
            os.mkdir("stl_color/models")
        if i % 10 == 0 or i == 49:
            torch.save(model_G.state_dict(), f"stl_color/models/gen_{i:03}.pytorch")
            torch.save(model_D.state_dict(), f"stl_color/models/dis_{i:03}.pytorch")

    writer.close()


if __name__ == "__main__":
    train()