import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rich.progress import track
from utils import log_some_examples, save_checkpoint, save_some_examples
from dataset import Sketch2ColorDataset
from generator import Generator
from discriminator import Discriminator
import hydra
from omegaconf import OmegaConf
import wandb

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg):
    print("Configurations:")
    print(OmegaConf.to_yaml(cfg))
    

    device = torch.device(cfg.processing.device)
    lr = cfg.training.lr

    discriminator = Discriminator(in_channels=3).to(device)
    generator = Generator(in_channels=3).to(device)

    disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    disc_scaler = torch.cuda.amp.GradScaler()
    gen_scaler = torch.cuda.amp.GradScaler()
    
    bce = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    train_dataset = Sketch2ColorDataset(root_dir=cfg.training.train_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.processing.batch_size,
        shuffle=True,
        num_workers=cfg.processing.num_workers,
    )

    val_dataset = Sketch2ColorDataset(root_dir=cfg.training.val_dir)
    val_loader = DataLoader(val_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=cfg.processing.num_workers)

    epochs = cfg.training.num_epochs

    experiment = wandb.init(project='pix2pix', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=cfg.processing.batch_size, learning_rate=lr, amp=True))
    global_step = 0

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
                L1 = l1_loss(y_fake, y) * cfg.training.l1_lambda
                G_loss = G_fake_loss + L1

            gen_optim.zero_grad()
            gen_scaler.scale(G_loss).backward()
            gen_scaler.step(gen_optim)
            gen_scaler.update()

            global_step += 1
            experiment.log({
                'discriminator_loss': disc_loss.item(),
                'generator_loss': G_loss.item(),
                'global_step': global_step,
                'epoch': epoch
            })

        real_score = torch.sigmoid(D_real).mean().item(),
        fake_score = torch.sigmoid(D_fake).mean().item(),
        colorized, sketches, generated_images = log_some_examples(generator, val_loader, device=device)
        
        experiment.log({ 
            'epoch_real_score(discriminator)': real_score,
            'epoch_fake_score(discriminator)': fake_score,
        })

        experiment.log({ 
            'original': wandb.Image(colorized),
            'sketches': wandb.Image(sketches),
            'generated': wandb.Image(generated_images),
        
        })
        
        if epoch % cfg.processing.save_model_every_n_epochs == 0:
            save_checkpoint(generator, gen_optim, filename=f"generator-epoch{epoch}.tar")
            save_checkpoint(discriminator, disc_optim, filename=f"discriminator-epoch{epoch}.tar")


if __name__ == "__main__":
    main()