# train.py for AWS AI and ML - Udacity Scholarship program

import argparse
import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from collections import OrderedDict

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print("Function called")
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = 2048
       
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        
        for param in model.features.parameters():
            param.requires_grad = False
      
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError("Unsupported architecture. Please choose from 'densenet121', 'resnet50', 'alexnet', or 'vgg16'.")

    criterion = nn.NLLLoss()
    
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Training Loss val: {running_loss/len(dataloaders['train']):.3f} | "
              f"Validation Loss val: {valid_loss/len(dataloaders['valid']):.3f} | "
              f"Validation Accuracy val: {accuracy/len(dataloaders['valid']):.3f}")

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'hidden_units': hidden_units
    }
    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"\nModel checkpoint saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('data_dir', type=str, help='Path')
    parser.add_argument('--save_dir', type=str, help='Directory')
    parser.add_argument('--arch', type=str, help='Architecture')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, help='hidden units')
    parser.add_argument('--epochs', type=int, help='training epochs')
    parser.add_argument('--gpu', action='store_true', help=' GPU for training')

    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)



