from generator import Generator
import torch
import cv2
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 

dir_ = "pix2pix-inference-samples"
if not os.path.exists(dir_): os.mkdir(dir_)

transforms = A.Compose(
    [   A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

generator = Generator()
gen_dict = torch.load("sketch-to-color-weights/generator-epoch95.tar")
weights = gen_dict["state_dict"]
generator.load_state_dict(weights)
generator.to(device=torch.device("cuda"))
generator.eval()


image = cv2.imread(f"{dir_}/sample.jpg")
transformed = transforms(image=image)
image = transformed["image"]
image = image[None,:,:,:].cuda().float()

with torch.no_grad():
    output = generator(image)

save_image(output *0.5 + 0.5, f"{dir_}generator_output.png")
