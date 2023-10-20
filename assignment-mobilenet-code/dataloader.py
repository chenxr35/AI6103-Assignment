import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms


class CustomDataset(data.Dataset):
    channel_mean = np.random.random((3))
    channel_std = np.random.random((3))

    def __init__(self, train_dataset, idxs, augment):

        normalize = transforms.Normalize(
            mean=list(CustomDataset.channel_mean),
            std=list(CustomDataset.channel_std),
        )

        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
            ])

        self.data = [transform(Image.fromarray(np.uint8(train_dataset.data[idx])).convert('RGB')) for idx in idxs]
        self.targets = np.array(train_dataset.targets)[idxs]
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def class_proportion(self, cls):
        return len(np.where(self.targets==cls)[0])/len(self.targets)


def random_train_val_split(train_dataset, valid_size, random_seed):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def calculate_mean_std(train_dataset, train_idx):
    train_data = np.array(train_dataset.data)[train_idx]
    N, H, W, C = train_data.shape[:4]
    train_data = train_data.reshape((N, C, -1))
    train_data = train_data / 255
    channel_mean = np.sum(np.mean(train_data, axis=2), axis=0) / N
    channel_std = np.sum(np.std(train_data, axis=2), axis=0) / N
    return channel_mean, channel_std


def get_train_valid_loader(dataset_dir, batch_size, augment, random_seed, save_images):
    train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=None, download=False)

    train_idx, valid_idx = random_train_val_split(train_dataset, 0.2, random_seed)

    channel_mean, channel_std = calculate_mean_std(train_dataset, train_idx)

    CustomDataset.channel_mean = channel_mean
    CustomDataset.channel_std = channel_std

    train_set = CustomDataset(train_dataset, train_idx, augment)

    valid_set = CustomDataset(train_dataset, valid_idx, False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        num_workers=2, pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, 
        num_workers=2, pin_memory=True,
    )

    return train_loader, valid_loader


def get_test_loader(dataset_dir, batch_size):
    
    test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=False)

    test_idx = [i for i in range(len(test_dataset))]

    test_set = CustomDataset(test_dataset, test_idx, False)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, 
        num_workers=2, pin_memory=True,
    )

    return test_loader