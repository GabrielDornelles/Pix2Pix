import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.progress import track

import config
from utils import save_checkpoint, save_some_examples
from dataset import Sketch2ColorDataset
from my_generator import Generator
from my_discriminator import Discriminator



torch.backends.cudnn.benchmark = True

discriminator = Discriminator(in_channels=3).to(config.DEVICE)
generator = Generator(in_channels=3).to(config.DEVICE)
disc_optim = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
gen_optim = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

train_dataset = Sketch2ColorDataset(root_dir=config.TRAIN_DIR)
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)

val_dataset = Sketch2ColorDataset(root_dir=config.VAL_DIR)
val_loader = DataLoader(val_dataset, 
    batch_size=16, 
    shuffle=False,
    num_workers=config.NUM_WORKERS)

gen_scaler = torch.cuda.amp.GradScaler()
disc_scaler = torch.cuda.amp.GradScaler()

device = config.DEVICE
epochs = config.NUM_EPOCHS


for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs}")
    

    for x,y in track(train_loader, description="Training..."):
        x = x.to(device)
        y = y.to(device)

        # Discriminator
        with torch.cuda.amp.autocast(): # Autocast makes training ~3x faster because of its mixed precision
            
            D_real = discriminator(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))

            y_fake = generator(x)
            D_fake = discriminator(x, y_fake.detach())

            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            disc_loss = (D_real_loss + D_fake_loss) / 2
        
        disc_optim.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_optim)
        disc_scaler.update()

        # Generator
        with torch.cuda.amp.autocast():
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        gen_optim.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(gen_optim)
        gen_scaler.update()

    real_score = torch.sigmoid(D_real).mean().item(),
    fake_score = torch.sigmoid(D_fake).mean().item(),
    print(f"Discriminator score for Real image: {real_score}")
    print(f"Discriminator score for Fake image: {fake_score}")
    save_some_examples(generator, val_loader, epoch, folder="evaluation")
    if config.SAVE_MODEL and epoch % 5 == 0:
        save_checkpoint(generator, gen_optim, filename=f"generator-epoch{epoch}.tar")
        save_checkpoint(discriminator, disc_optim, filename=f"discriminator-epoch{epoch}.tar")
        
