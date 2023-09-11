import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import config as cfg

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=cfg.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(cfg.image_size),
                               transforms.CenterCrop(cfg.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                         shuffle=True, num_workers=cfg.workers)
