# This script is used to calculate 
# the mean and std of the image dataset intensity
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image

from torchvision import datasets

torch.cuda.empty_cache()

class GrayscaleSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'raw')

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_filename = os.listdir(self.image_dir)[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image



# path of data 
root_dir = 'C:/Users/yang.xin1022/Desktop/train_test2/train'


# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create an instance of the GrayscaleSegmentationDataset
dataset = GrayscaleSegmentationDataset(root_dir, transform=transform)

# Create a data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

# Initialize variables to accumulate the sum and sum of squares
mean = torch.zeros(1)
std = torch.zeros(1)
total_images = 0

# Iterate over the dataset and calculate mean and std
for images in data_loader:
    batch_size = images.size(0)
    height = images.size(2)
    width = images.size(3)

    # Flatten the images for mean and std calculation
    images = images.view(batch_size, -1)

    # Calculate sum and sum of squares
    mean += images.mean(1).sum(0)
    std += images.std(1).sum(0)

    total_images += batch_size

# Calculate the overall mean and std
mean /= total_images
std /= total_images

print(f'Mean: {mean.tolist()}')
print(f'Std: {std.tolist()}')