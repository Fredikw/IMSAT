import os

from torch import squeeze
from torch.utils import data
from torchvision import transforms

from PIL import ImageOps, Image

from sklearn.model_selection import train_test_split

TRAIN_PATHS:  list = []
TRAIN_LABELS: list = []
TEST_PATHS:   list = []
TEST_LABELS:  list = []

# max_width, max_height
MAX_DIMENSION: tuple = () # (424, 428)


"""
Data Preprocessing

"""
class NDSBDataset(data.Dataset):
    def __init__(self, train: bool = True, augment_data: bool = False):
        self.train = train
        # Augmentation should not be applied for testing
        self.augment_data = augment_data and train

        if self.train:
            self.labels = TRAIN_LABELS
            self.paths  = TRAIN_PATHS
        else:
            self.labels = TEST_LABELS
            self.paths  = TEST_PATHS

        self.transform_list = transforms.Compose([
            transforms.RandomRotation(degrees=180, fill=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter()
        ])

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.paths[index])
        # Resize image to the largest image in the dataset
        img = ImageOps.pad(image=img, size=MAX_DIMENSION, color=255)
        img = transforms.ToTensor()(img)

        label = self.labels[index]

        if self.augment_data:
            img_trans = self.transform_list(img)
            return (squeeze(img.view(1, -1)), squeeze(img_trans.view(1, -1))), label

        return squeeze(img.view(1, -1)), label
    
    def __len__(self):
        return len(self.paths)

"""
Utility Functions

"""

# Split dataset into random train and test subsets.
def init_dataset(data_dir: str) -> tuple:
    """
    Split dataset into random train and test subset.

    Args:
        data_dir (str): root directory path of the dataset.
    """

    ndsb_labels    = []
    ndsb_img_paths = []

    # Read label and file paths
    for label, label_path in enumerate(sorted(os.listdir(data_dir))):
        for sample in os.listdir(os.path.join(data_dir, label_path)):
            ndsb_labels.append(label)
            ndsb_img_paths.append(os.path.join(data_dir, label_path, sample))


    # Split dataset into training and test set

    train_paths, test_paths, train_labels, test_labels = \
        train_test_split(ndsb_img_paths, ndsb_labels)
    
    return train_paths, test_paths, train_labels, test_labels

def find_max_dimension() -> tuple:

    max_width:  int = 0
    max_height: int = 0

    ndsb_img_paths = TRAIN_PATHS + TEST_PATHS

    for image_path in ndsb_img_paths:
        with Image.open(image_path) as img:
            width, height = img.size
            
            max_width  = max(max_width, width)
            max_height = max(max_height, height)

    return max_width, max_height

"""
MNIST dataset for testing

"""

from torchvision.datasets import MNIST

class MNISTDataset(data.Dataset):
    def __init__(self, train=True, augment_data=False):
        # super().__init__()

        self.train        = train
        self.augment_data = augment_data

        # Load the MNIST dataset
        self.mnist = MNIST(
            root='./data',
            train=self.train,
            download=True,
            transform=transforms.ToTensor())

        self.transform_list = transforms.Compose([
            transforms.RandomRotation(degrees=30)
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        if self.augment_data:
            img_aug = self.transform_list(self.mnist[index][0])
            # Return the augmented image, original image and label at the given index
            return (squeeze(self.mnist[index][0].view(-1, 28*28)), squeeze(img_aug.view(-1, 28*28))), self.mnist[index][1]

        # Return the original image and label at the given index
        return  squeeze(self.mnist[index][0].view(-1,28*28)), self.mnist[index][1]


if __name__ == '__main__':
    pass
else:
    TRAIN_PATHS, TEST_PATHS, TRAIN_LABELS, TEST_LABELS = init_dataset("./data/NDSB/train")
    MAX_DIMENSION = (428, 428) # find_max_dimension()