import torch
import config as cfg
import torch.nn as nn
import torch.optim as optim
from data import dataloader
from model import get_models

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()


def train(netG, netD, dataloader, num_epochs):
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    iters = 0
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(cfg.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), cfg.real_label,
                               dtype=torch.float, device=cfg.device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, cfg.nz, 1, 1, device=cfg.device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(cfg.fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(cfg.real_label)
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
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(cfg.fixed_noise).detach().cpu()
                img_list.append(cfg.vutils.make_grid(
                    fake, padding=2, normalize=True))

            iters += 1


netG, netD = get_models()
train(netG, netD, dataloader, cfg.num_epochs)
