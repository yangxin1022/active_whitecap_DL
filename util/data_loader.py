import PIL
import os
from torch.utils.data import Dataset, DataLoader
import torchvision

# Active whitecap fraction dataset
class ActiveWFDataset(Dataset):
    """ A customized dataset for active whitecaps image segementation"""
    def __init__(self, features_dir, labels_dir):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([256,256]),
            torchvision.transforms.Normalize(
                mean=0.3374, std=0.1843)
        ])
        self.feature_folder = features_dir 
        self.labels_folder = labels_dir
        features = [PIL.Image.open(
            os.path.join(self.feature_folder, fname)).convert('L')
                    for fname in os.listdir(self.labels_folder)]
        labels = [PIL.Image.open(
            os.path.join(self.labels_folder, fname)).convert('L')
                    for fname in os.listdir(self.labels_folder)]
        # the filename of features and labels should be the same
        self.features = [self.transform(self.normalize_image(feature))
                         for feature in features]
        self.labels = [self.normalize_image(label)
                         for label in labels]
 
        
    def normalize_image(self, img):
        scaletransform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize([256,256]),
             torchvision.transforms.PILToTensor()])
        img = scaletransform(img)
        return img.float() / 255
        
    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]
        return (feature, label)
    
    def __len__(self):
        return len(self.features)