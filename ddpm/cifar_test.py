
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def train_CNN(config, device, train_loader, test_loader):
    """
    Trains a basic ResNet18 on the CIFAR10 data, just to check.
    """
    net = resnet18(num_classes=10).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])


    for epoch in range(config['num_epochs']):  # loop over the dataset multiple times

        # Training
        net.train()
        running_loss = 0.0
        correct = 0; total=0
        for inputs, labels in train_loader:
            # Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = net(inputs)
            # Training logic
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Training loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch}: training loss {running_loss:.4f}; accuracy {correct/total:.4f}")
        
        # Validation
        net.eval()
        running_loss = 0.0
        correct = 0; total=0
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move inputs and labels to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                # Validation loss
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"        validation loss {running_loss:.4f}; accuracy {correct/total:.4f}")

        # Adjust learning rate
        scheduler.step()

    torch.save(net.state_dict(), './checkpoints/cnn_cifar.pth')