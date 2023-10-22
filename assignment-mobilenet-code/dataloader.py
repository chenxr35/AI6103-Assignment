import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(dataset_dir, batch_size, augment, random_seed, save_images):

    normalize = transforms.Normalize(
        mean=[0.5179, 0.4660, 0.4579],
        std=[0.1884, 0.2026, 0.1855],
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.cifar.CIFAR100(
        root='cifar100', train=True,
        transform=train_transform, download=True
    )

    valid_dataset = datasets.cifar.CIFAR100(
        root='cifar100', train=True,
        transform=valid_transform, download=True
    )

    valid_size = 0.2
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=2, pin_memory=True,
    )

    return train_loader, valid_loader


def get_test_loader(dataset_dir, batch_size):

    normalize = transforms.Normalize(
        mean=[0.5179, 0.4660, 0.4579],
        std=[0.1884, 0.2026, 0.1855],
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    test_dataset = datasets.cifar.CIFAR100(
        root='cifar100', train=False,
        transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=2, pin_memory=True,
    )

    return test_loader