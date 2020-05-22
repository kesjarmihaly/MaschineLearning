import torch
import torchvision

from torch import nn
from torch.utils.tensorboard import SummaryWriter


from srgan import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss

from utils import load_datasets, psnr, ssim

from tqdm import tqdm
import statistics
import os
import argparse

def get_psnr(fake, real):
    cpu = torch.device("cpu")

    psnr_list = []
    for i in range(len(fake)):
        np_fake = fake[i].to(cpu).detach().clone().numpy().transpose([1, 2, 0])
        np_real = real[i].to(cpu).detach().clone().numpy().transpose([1, 2, 0])
        psnr_list.append(psnr(np_fake, np_real))
    return statistics.mean(psnr_list)

def get_ssim(fake, real):
    cpu = torch.device("cpu")

    ssim_list = []
    for i in range(len(fake)):
        np_fake = fake[i].to(cpu).detach().clone().numpy().transpose([1, 2, 0])
        np_real = real[i].to(cpu).detach().clone().numpy().transpose([1, 2, 0])
        ssim_list.append(ssim(np_fake, np_real))
    return statistics.mean(ssim_list)

def pre_train(x_train, x_test, pre_epochs, upscale_factor, device, save_image_path, save_model_path):

    writer = SummaryWriter()

    model_G = Generator(upscale_factor).to(device)

    optimizer_G = torch.optim.Adam(model_G.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))

    mae_loss = nn.L1Loss()
    best_loss = 99999
    best_model_name = ''

    result = {}
    result["pre_train/l1_loss_G"] = []
    result["pre_test/l1_loss_G"] = []
    result["pre_test/psnr"] = []
    result["pre_test/ssim"] = []

    for epoch in range(pre_epochs):
        print("train step: epoch {}".format(epoch))
        train_loss_G = []
        for (input_img, real_img), _ in tqdm(x_train):

            input_img, real_img = input_img.to(device), real_img.to(device)

            optimizer_G.zero_grad()

            fake_img = model_G(input_img)
            g_loss = mae_loss(real_img, fake_img)
            g_loss.backward()
            optimizer_G.step()

            train_loss_G.append(g_loss.item())

        result["pre_train/l1_loss_G"].append(statistics.mean(train_loss_G))
        writer.add_scalar('pre_train/l1_loss_G', result["pre_train/l1_loss_G"][-1], epoch)

        if epoch % 5 == 0 or epoch == pre_epochs - 1:
            with torch.no_grad():
                print("test step")
                test_loss_G = []
                test_psnr = []
                test_ssim = []
                for (input_img, real_img), _ in tqdm(x_test):

                    input_img, real_img = input_img.to(device), real_img.to(device)

                    fake_img = model_G(input_img)
                    g_loss = mae_loss(real_img, fake_img)

                    test_loss_G.append(g_loss.item())

                    test_psnr.append(get_psnr(fake_img, real_img))
                    test_ssim.append(get_ssim(fake_img, real_img))


                total_test_loss_G = statistics.mean(test_loss_G)
                result["pre_test/l1_loss_G"].append(total_test_loss_G)
                writer.add_scalar('pre_test/l1_loss_G', result["pre_test/l1_loss_G"][-1], epoch)

                result["pre_test/psnr"].append(statistics.mean(test_psnr))
                result["pre_test/ssim"].append(statistics.mean(test_ssim))
                writer.add_scalar('pre_test/psnr', result["pre_test/psnr"][-1], epoch)
                writer.add_scalar('pre_test/ssim', result["pre_test/ssim"][-1], epoch)

                torchvision.utils.save_image(fake_img[:32],
                                             os.path.join(save_image_path, f"fake_epoch_{epoch:03}.png"),
                                             range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(real_img[:32],
                                             os.path.join(save_image_path, f"real_epoch_{epoch:03}.png"),
                                             range=(-1.0, 1.0), normalize=True)

            if best_loss > total_test_loss_G:
                best_loss = total_test_loss_G
                best_model_name = os.path.join(save_model_path, f"pre_gen_{epoch:03}.pytorch")
                print("save model ==>> {}".format(best_model_name))

                torch.save(model_G.state_dict(), best_model_name)

    writer.close()

    return best_model_name


def train(x_train, x_test, epochs, upscale_factor, device, pre_model, save_image_path, save_model_path):

    writer = SummaryWriter()

    model_G = Generator(upscale_factor)
    model_G.load_state_dict(torch.load(pre_model))
    model_G = model_G.to(device)

    optimizer_G = torch.optim.Adam(model_G.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))

    model_D = Discriminator()
    model_D = model_D.to(device)

    optimizer_D = torch.optim.Adam(model_D.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))

    loss_G = GeneratorLoss().to(device)
    loss_D = DiscriminatorLoss().to(device)

    result = {}
    result["train/loss_G"] = []
    result["train/loss_D"] = []
    result["test/loss_G"] = []
    result["test/loss_D"] = []
    result["test/psnr"] = []
    result["test/ssim"] = []

    for epoch in range(epochs):
        print("train step: epoch {}".format(epoch))
        train_loss_G, train_loss_D = [], []

        for (input_img, real_img), _ in tqdm(x_train):

            input_img, real_img = input_img.to(device), real_img.to(device)

            fake_img = model_G(input_img)
            fake_img_tensor = fake_img.detach()

            # Update D
            optimizer_D.zero_grad()
            D_out_real = model_D(real_img)
            D_out_fake = model_D(fake_img_tensor)
            d_loss = loss_D(D_out_real, D_out_fake)
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            train_loss_D.append(d_loss.item())

            # Update G
            optimizer_G.zero_grad()
            g_loss = loss_G(D_out_fake, real_img, fake_img)
            g_loss.backward(retain_graph=True)
            optimizer_G.step()

            train_loss_G.append(g_loss.item())


        result["train/loss_G"].append(statistics.mean(train_loss_G))
        result["train/loss_D"].append(statistics.mean(train_loss_D))

        writer.add_scalar('train/loss_G', result["train/loss_G"][-1], epoch)
        writer.add_scalar('train/loss_D', result["train/loss_D"][-1], epoch)

        if epoch % 5 == 0 or epoch == epochs - 1:

            with torch.no_grad():

                print("test step")
                test_loss_G, test_loss_D = [], []
                test_psnr, test_ssim = [], []
                for (input_img, real_img), _ in tqdm(x_test):

                    input_img, real_img = input_img.to(device), real_img.to(device)

                    fake_img = model_G(input_img)

                    D_out_real = model_D(real_img)
                    D_out_fake = model_D(fake_img)
                    d_loss = loss_D(D_out_real, D_out_fake)
                    test_loss_D.append(d_loss.item())

                    g_loss = loss_G(D_out_fake, real_img, fake_img)
                    test_loss_G.append(g_loss.item())

                    test_psnr.append(get_psnr(fake_img, real_img))
                    test_ssim.append(get_ssim(fake_img, real_img))

                result["test/loss_G"].append(statistics.mean(test_loss_G))
                result["test/loss_D"].append(statistics.mean(test_loss_D))
                result["test/psnr"].append(statistics.mean(test_psnr))
                result["test/ssim"].append(statistics.mean(test_ssim))

                writer.add_scalar('test/loss_G', result["test/loss_G"][-1], epoch)
                writer.add_scalar('test/loss_D', result["test/loss_D"][-1], epoch)
                writer.add_scalar('test/psnr', result["test/psnr"][-1], epoch)
                writer.add_scalar('test/ssim', result["test/ssim"][-1], epoch)

                torchvision.utils.save_image(fake_img[:32],
                                             os.path.join(save_image_path, f"fake_epoch_{epoch:03}.png"),
                                             range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(real_img[:32],
                                             os.path.join(save_image_path, f"real_epoch_{epoch:03}.png"),
                                             range=(-1.0, 1.0), normalize=True)


            if epoch % 10 == 0 or epoch == epochs - 1:
                torch.save(model_G.state_dict(), os.path.join(save_model_path, f"gen_{epoch:03}.pytorch"))
                torch.save(model_D.state_dict(), os.path.join(save_model_path, f"dis_{epoch:03}.pytorch"))

    writer.close()

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_model_path', type=str, default="sr_rnn/model")

    parser.add_argument('--pre_image_path', type=str, default="sr_rnn/images")

    parser.add_argument('--model_path', type=str, default="sr_gan/model")

    parser.add_argument('--image_path', type=str, default="sr_gan/images")

    parser.add_argument('--pre_epochs', type=int, default=50)

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--upscale_factor', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    return args

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test = load_datasets(args.upscale_factor, args.batch_size)

    check_dir(args.pre_image_path)
    check_dir(args.pre_model_path)

    pre_model_name = pre_train(
                        x_train=x_train,
                        x_test=x_test,
                        pre_epochs=args.pre_epochs,
                        upscale_factor=args.upscale_factor,
                        device=device,
                        save_image_path=args.pre_image_path,
                        save_model_path=args.pre_model_path,
                    )

    check_dir(args.image_path)
    check_dir(args.model_path)

    train(
        x_train=x_train,
        x_test=x_test,
        epochs=args.epochs,
        upscale_factor=args.upscale_factor,
        device=device,
        pre_model=pre_model_name,
        save_image_path=args.image_path,
        save_model_path=args.model_path,
    )


if __name__ == '__main__':
    args = arg_parser()
    main(args)