from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from tqdm import tqdm

from model import Generator, Discriminator

# Set random seed for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_data(args):

    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers)

    return dataloader

def train(args):

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    dataloader = load_data(args)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # Create the generator
    netG = Generator(args.ngpu, args.nc, args.nz, args.ngf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(args.ngpu, args.nc, args.ndf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (args.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(args.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    losses_for_csv = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.num_epochs):
        print("Epoch: {}".format(epoch))
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, 0)):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            losses_for_csv.append([iters, errG.item(), errD.item()])

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                vutils.save_image(fake,
                                 os.path.join(args.save_sample_path, f"fake_iter_{iters:03}.png"),
                                 nrow=8, range=(-1.0, 1.0), normalize=True)

            iters += 1

        if epoch % 10 == 0 or args.num_epochs - 1 == 0:
            torch.save(netG.state_dict(),
                       os.path.join(args.save_model_path, f"gen_{epoch:03}.pt"))
            torch.save(netD.state_dict(),
                       os.path.join(args.save_model_path, f"dis_{epoch:03}.pt"))

    save_file_name = os.path.join(args.save_image_path, "Gen_Dis_loss.png")
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    print("Saving ==>> {}".format(save_file_name))
    plt.savefig(save_file_name)

    save_file_name = os.path.join(args.save_image_path, "Gen_animation.gif")
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    print("Saving ==>> {}".format(save_file_name))
    ani.save(save_file_name, writer='imagemagick', fps=4)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    save_file_name = os.path.join(args.save_image_path, "Gen_img.png")
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    print("Saving ==>> {}".format(save_file_name))
    plt.savefig(save_file_name)

    save_file_name = os.path.join(args.save_csv_path, "Gen_Dis_loss.csv")
    df = pd.DataFrame(losses_for_csv, columns=['Iteration', 'Generator Loss', 'Discriminator Loss'])
    print("Saving ==>> {}".format(save_file_name))
    df.to_csv(save_file_name, index=False)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default="./celeba/")
    parser.add_argument('--save_model_path', type=str, default="./model/")
    parser.add_argument('--save_csv_path', type=str, default="./csv/")
    parser.add_argument('--save_sample_path', type=str, default="./sample/")
    parser.add_argument('--save_image_path', type=str, default="./image/")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nz', type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument('--nc', type=int, default=3, help="Number of channels in the training images")
    parser.add_argument('--ngf', type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument('--ndf', type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument('--ngpu', type=int, default=1, help="Number of GPUs available. Use 0 for CPU mode")
    parser.add_argument('--workers', type=int, default=2, help="Number of workers for dataloader")

    args = parser.parse_args()
    return args

def main(args):
    check_dir(args.save_model_path)
    check_dir(args.save_csv_path)
    check_dir(args.save_sample_path)
    check_dir(args.save_image_path)

    train(args)

if __name__ == '__main__':
    args = arg_parser()
    main(args)
