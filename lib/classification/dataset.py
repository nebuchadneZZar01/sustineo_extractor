import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image

class PlotDataset(Dataset):
    """Object defining the plot dataset to be used.

    Keyword Arguments:
        - annotations_file -- Path to the csv file with annotations
        - root_dir -- Directory containing all the images
        - transform (optional) -- Optional transform to be applied on a sample
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label