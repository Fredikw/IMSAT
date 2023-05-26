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
MAX_DIMENSIONS: tuple = () # (424, 428)
DIMENSIONS: tuple = ()

"""
Dataset class for National Data Science Bowl plankton dataset

https://www.kaggle.com/competitions/datasciencebowl/data
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
        
        # TODO Consider different compositions
        # self.transform_list = transforms.RandomChoice([])
        # self.transform_list = transforms.RandomApply([])
        self.transform_list = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180, fill=1),
            transforms.RandomResizedCrop(size=DIMENSIONS, scale=(0.80, 1.)),
            transforms.ColorJitter(brightness=0.2),
            transforms.GaussianBlur(kernel_size=21)
            # transforms.RandomAdjustSharpness(sharpness_factor=2)
            # transforms.RandomAffine(degrees=180, fill=1)
        ])

    def __getitem__(self, index):
        # Load image
        img = Image.open(self.paths[index])
        # Resize image to the largest image in the dataset
        img = ImageOps.pad(image=img, size=DIMENSIONS, color=255)
        img = transforms.ToTensor()(img)

        label = self.labels[index]

        return img, label
    
    def __len__(self):
        return len(self.paths)

"""
MNIST dataset for testing purposes

"""

from torchvision.datasets import MNIST

class MNISTDataset(data.Dataset):
    def __init__(self, train=True, augment_data=False):
        # super().__init__()

        self.train        = train
        # Augmentation should not be applied for testing
        self.augment_data = augment_data and train

        # Load the MNIST dataset
        self.mnist = MNIST(
            root='./datasets',
            train=self.train,
            download=True,
            transform=transforms.ToTensor())

        self.transform_list = transforms.Compose([
            transforms.RandomRotation(degrees=30)
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        # Return the original image and label at the given index
        return  self.mnist[index][0], self.mnist[index][1]

"""
Utility Functions

"""

def init_dataset(data_dir: str, subset: bool=False) -> tuple:
    """
    Split dataset into random train and test subset.

    Args:
        data_dir (str): root directory path of the dataset.
        subset (bool): If True, only consider a subset of classes.
    """
    subset_classes = ['acantharia_protist',
                      'chordate_type1',
                      'copepod_calanoid_eucalanus',
                      'copepod_cyclopoid_copilia',
                      'ctenophore_cestid',
                      'ctenophore_lobate',
                      'diatom_chain_string',
                      'echinoderm_larva_seastar_brachiolaria',
                      'hydromedusae_haliscera',
                      'radiolarian_chain']

    ndsb_labels    = []
    ndsb_img_paths = []

    # Read label and file paths
    for label, label_path in enumerate(sorted(os.listdir(data_dir))):
        # If subset is True, only consider classes in subset_classes
        if subset and label_path not in subset_classes:
            continue
        for sample in os.listdir(os.path.join(data_dir, label_path)):
            ndsb_labels.append(label) if not subset else ndsb_labels.append(subset_classes.index(label_path))
            ndsb_img_paths.append(os.path.join(data_dir, label_path, sample))

    # Split dataset into training and test set
    train_paths, test_paths, train_labels, test_labels = \
        train_test_split(ndsb_img_paths, ndsb_labels)
    
    return train_paths, test_paths, train_labels, test_labels

if __name__ == '__main__':
    pass
else:
    TRAIN_PATHS, TEST_PATHS, TRAIN_LABELS, TEST_LABELS = init_dataset("./datasets/NDSB/train", subset=True)
    # MAX_DIMENSION = (428, 428) # find_max_dimension()
    DIMENSIONS = (299, 299)