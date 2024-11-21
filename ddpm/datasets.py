import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10(batch_size = 64, testset=False):
    
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if testset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    if testset:
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        labels = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if testset:
        return train_data, train_loader, test_data, test_loader, labels
    else:
        return train_data, train_loader