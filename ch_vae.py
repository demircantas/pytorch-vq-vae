import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained VQ-VAE-2 model
model_path = "256x256_diffusion_uncond.pt"
vqvae = torch.load(model_path)["model"]
vqvae.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

# Load and preprocess input image
image_path = "input_image.jpg"
image = transform(Image.open(image_path)).unsqueeze(0)

# Encode the input image
with torch.no_grad():
    _, _, encoding = vqvae.encode(image)

# Decode the latent vector
with torch.no_grad():
    decoded_image = vqvae.decode(encoding)

# Save the decoded image
output_image_path = "output_image.jpg"
torchvision.utils.save_image(decoded_image, output_image_path)


import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained VQ-VAE-2 model
model_path = "256x256_diffusion_uncond.pt"
checkpoint = torch.load(model_path)
vqvae = checkpoint["model"] if "model" in checkpoint else checkpoint

# Set the model in evaluation mode
vqvae.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

# Load and preprocess input image
image_path = "input_image.jpg"
image = transform(Image.open(image_path)).unsqueeze(0)

# Encode the input image
with torch.no_grad():
    encoding = vqvae.encode(image)

print(f'Encoding shape: {encoding.shape}')

# Decode the latent vector
with torch.no_grad():
    decoded_image = vqvae.decode(encoding)

# Save the decoded image
output_image_path = "output_image.jpg"
torchvision.utils.save_image(decoded_image, output_image_path)
