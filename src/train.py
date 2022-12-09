import os
import sys
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import ViT
from utils import load_datasets

# CUDNN 에서 최상의 알고리즘을 찾는 기능, 모델에 따라 느릴 수도 있음
'''https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936'''
cudnn.benchmark = True

# using epoch, batch
# save model
# save best model
def train(model, criterion, optimizer, scheduler, save_dir, num_epochs=25):
    """https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html 코드 수정하여 사용 중"""
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    pbar = tqdm(range(num_epochs))

    for epoch in pbar:

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # batch
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # feed forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # epoch loss
            globals()[f'{phase}_loss'] = epoch_loss / dataset_sizes[phase]
            globals()[f'{phase}_acc'] = epoch_corrects.double() / dataset_sizes[phase]

            # deepcopy best models' weight
            if phase == 'val':
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        # progressbar log
        pbar.set_description(f'[{model._get_name()}]')
        pbar.set_postfix({'train loss': round(train_loss, 2),
                          'acc': torch.round(train_acc.detach(), decimals=2).numpy(),
                          'val loss': round(val_loss, 2),
                          'acc': torch.round(val_acc.detach(), decimals=2).numpy(),})

        # save model
        if epoch % 50 == 0 or epoch == (num_epochs - 1):
            torch.save({"weight": model.state_dict(),
                        "epoch": epoch,
                         "loss": val_loss,
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                        },
                       os.path.join(save_dir, f"{model._get_name()}_{epoch}.pth"),
            )

    time_elapsed = time.time() - start_time
    print('=' * 30)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Epoch: {best_epoch} Acc: {best_acc:4f}')
    print('=' * 30)

    # save best model to pth
    model.load_state_dict(best_model_wts)
    torch.save({"weight": model.state_dict()},
               os.path.join(save_dir, f"{model._get_name()}_best.pth"),)
    return model

if __name__ == '__main__':

    # datasets path
    dataset_name = "hymenoptera_data"

    # setting parameter, model for model train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ViT(n_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # setting dataloader
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    '''os.getcwd().split(os.path.sep + 'src')[0] -> os.path.abspath(os.path.pardir)'''
    data_dir = os.path.join(os.getcwd().split(os.path.sep + 'src')[0], "datasets", dataset_name)
    dataloaders, dataset_sizes, class_names = load_datasets(data_dir=data_dir,
                                                            data_transforms=data_transforms,
                                                            batch_size=64,
                                                            num_workers=0)

    # train
    save_dir = os.path.join(os.getcwd().split(os.path.sep + 'src')[0], "checkpoints")
    train(model, criterion, optimizer, scheduler, save_dir, num_epochs=5)
