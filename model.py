import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms, models
from imagepathloader import ImagePathLoader
import torch.nn.functional as F
import pandas as pd


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


    dataset_train = datasets.ImageFolder(root='data\\train', transform=preprocessing_train)
    dataset_valid = datasets.ImageFolder(root='data\\valid', transform=preprocessing_test)
    #dataset_test = datasets.ImageFolder(root='data/test', transform=preprocessing_test)
    dataset_test = ImagePathLoader(root='data\\test', transform=preprocessing_test)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=64, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)

    #batch = next(iter(dataloader_train))
    #print(batch)
    print(dataset_train.classes)
    #print(dataset_train.class_to_idx["melanoma"])

    return dataloader_train, dataloader_valid, dataloader_test


def create_model():
    #model = models.resnet50(pretrained=True)
    model = models.inception_v3(pretrained=True, aux_logits=False)
    print(model)

    for param in model.parameters():
        param.requires_grad_(False)

    model.fc = nn.Linear(in_features=2048, out_features=3)
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
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
            print("Validation loss: {0} accuracy: {1}".format(valid_loss/len(dataloader_valid.dataset), accuracy/len(dataloader_valid.dataset)))

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)


def test_model(model, dataloader_test, report_path):

    df = pd.DataFrame(columns=["Id", "task_1", "task_2"])
    
    model.eval()
    with torch.no_grad():
        accuracy = 0
        for X, y, paths in dataloader_test:
            y_hat = model(X)
            
            probs = F.softmax(y_hat, dim=1)
            class_ids = y_hat.argmax(dim=1)
            eq = class_ids == y
            accuracy += eq.sum()

            melanoma_prob = probs[:, 0]
            keratosis_prob = probs[:, 2]

            df_part = pd.DataFrame({"Id": paths, 
                                    "task_1": melanoma_prob.numpy(),
                                    "task_2": keratosis_prob.numpy()})

            df = df.append(df_part, ignore_index=True)

        df.to_csv(report_path, index=False)
        print("Test accuracy: ", accuracy.item()/len(dataloader_test.dataset))



#input_size = 224
input_size = 299 # inception-v3 expects 299x299 images
dataloader_train, dataloader_valid, dataloader_test = create_datasets(input_size)
model, optimizer, criterion = create_model()
train_model(model, optimizer, criterion, dataloader_train, dataloader_valid, 5, "model_checkpoint2.pt")
##model = torch.load("model_checkpoint.pt")
model.load_state_dict(torch.load("model_checkpoint2.pt"))
test_model(model, dataloader_test, "predictions2.csv")




