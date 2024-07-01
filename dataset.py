import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val', specifying the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Populate image paths and labels
        for label, class_dir in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_name in os.listdir(class_dir_path):
                    self.image_paths.append(os.path.join(class_dir_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Usage example:
# Define a transform to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

# Create the dataset
imagenet_dataset = ImageNetDataset(root_dir='/path/to/imagenet', split='train', transform=transform)

# Create a DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(imagenet_dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through the dataset
for images, labels in dataloader:
    # Your training or validation code here
    pass
