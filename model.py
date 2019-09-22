import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F



def create_datasets(input_size):

    preprocessing_train = transforms.Compose([
                                            transforms.RandomRotation(20),
                                            transforms.RandomHorizontalFlip(0.3),
                                            transforms.RandomResizedCrop(input_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                            ])

    preprocessing_test = transforms.Compose([
                                            transforms.Resize(input_size),
                                            transforms.CenterCrop(input_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])


    dataset_train = datasets.ImageFolder(root='data/train', transform=preprocessing_train)
    dataset_valid = datasets.ImageFolder(root='data/valid', transform=preprocessing_test)
    dataset_test = datasets.ImageFolder(root='data/test', transform=preprocessing_test)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=64, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

    #batch = next(iter(dataloader_train))
    #print(batch)
    #print(dataset_train.classes)
    #print(dataset_train.class_to_idx["melanoma"])

    return dataloader_train, dataloader_valid, dataloader_test


def create_model():
    model = models.resnet50(pretrained=True)
    print(model)

    for param in model.parameters():
        param.requires_grad_(False)

    model.fc = nn.Linear(in_features=2048, out_features=3)
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    return model, optimizer, criterion

def train_model(model, optimizer, criterion, dataloader_train, dataloader_valid, n_epochs, save_path):
        
    for epoch in range(1, n_epochs+1):
        training_loss = 0
        model.train()
        for X, y in dataloader_train:
            y_hat = model(X)        
            loss = criterion(y_hat, y)
            training_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print("Epoch %d: training loss: %f" % (epoch, training_loss/len(dataloader_train.dataset)))
        print("Epoch {0}: training loss: {1}".format(epoch, training_loss/len(dataloader_train.dataset)))

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            accuracy = 0
            best_loss = np.Inf
            for X, y in dataloader_valid:
                y_hat = model(X)
                loss = criterion(y_hat, y)
                valid_loss += loss.item()

                eq = y_hat.argmax(dim=1) == y
                accuracy += eq.sum().item()

            #print(f"Validation loss: {valid_loss/len(dataloader_valid.dataset)} accuracy: {accuracy/len(dataloader_valid.dataset)}")
            #print("Validation loss: %f accuracy: %f" % (valid_loss/len(dataloader_valid.dataset), accuracy/len(dataloader_valid.dataset)))
            print("Validation loss: {0} accuracy: {1}".format(valid_loss/len(dataloader_valid.dataset), accuracy/len(dataloader_valid.dataset)))

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)


def test_model(model, dataloader_test):
    model.eval()

    with torch.no_grad():
        for X, y in dataloader_test:
            y_hat = model(X)
            prob = F.softmax(y_hat, dim=1)


input_size = 224
dataloader_train, dataloader_valid, dataloader_test = create_datasets(input_size)
model, optimizer, criterion = create_model()
train_model(model, optimizer, criterion, dataloader_train, dataloader_valid, 3, "model_checkpoint.pt")
#model = torch.load("model_checkpoint.pt")
model.load_state_dict(torch.load("model_checkpoint.pt"))
test_model(model, dataloader_test)



