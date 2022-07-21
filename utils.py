import torch
import config
from torchvision.utils import save_image
import numpy as np


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        print(y_fake.shape)
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def log_some_examples(gen, val_loader, device):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        y_fake = y_fake.detach().cpu().numpy().transpose([0,2,3,1])

    half = int(x.shape[0] / 2)
    # concat originals
    original_images = y.detach().cpu().numpy().transpose([0,2,3,1])
    upper = np.hstack(original_images[:half])
    lower = np.hstack(original_images[half:])
    images = np.vstack((upper,lower))

    # Concat sketches
    sketch_images = x.detach().cpu().numpy().transpose([0,2,3,1])
    upper = np.hstack(sketch_images[:half])
    lower = np.hstack(sketch_images[half:])
    sketches = np.vstack((upper,lower))

    # Concat generated images
    upper = np.hstack(y_fake[:half])
    lower = np.hstack(y_fake[half:])
    generator_images = np.vstack((upper,lower))

    return images, sketches, generator_images


def save_checkpoint(model, optimizer, filename="my_checkpoint.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])