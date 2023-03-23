from torch import squeeze
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST


"""
Data Preprocessing

"""

class MNISTDataset(data.Dataset):
    def __init__(self, train=True, augment=False):
        # super().__init__()

        self.train = train
        self.augment = augment

        # Load the MNIST dataset
        self.mnist = MNIST(
            root='./data',
            train=self.train,
            download=True,
            transform=transforms.ToTensor())

        # Apply data augmentation if requested
        if self.augment:
            # Define the data augmentation transforms to apply
            augmentations = transforms.Compose([
                            transforms.RandomRotation(degrees=15),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomCrop((28, 28), padding=0),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])

            # Create a new dataset by applying the transforms to the original dataset
            self.mnist_augmented = [(augmentations(img), label) for img, label in self.mnist]


    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        if self.augment:
            # TODO needs to be squeezed
            # Return the augmented image, original image and label at the given index
            return (squeeze(self.mnist[index][0].view(-1, 28*28)), squeeze(self.mnist_augmented[index][0].view(-1, 28*28))), self.mnist[index][1]
        else:
            # Return the original image and label at the given index
            return  squeeze(self.mnist[index][0].view(-1,28*28)), self.mnist[index][1]