
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from tqdm import tqdm
import albumentations as A
import cv2    
from albumentations.pytorch import ToTensorV2 as ToTensor
import time
from torch.amp import GradScaler
from torchvision import models
from efficientnet_pytorch import EfficientNet
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


albumentations_transforms=A.Compose([A.HorizontalFlip(p=.5),
                           A.ShiftScaleRotate(rotate_limit=2,value=0,
                                              border_mode=cv2.BORDER_CONSTANT, p=.5),

                           A.OneOf(
                                   [A.CLAHE(clip_limit=2),
                                    A.RandomBrightnessContrast(),
                                   ],p=.5),
                           A.GaussNoise(var_limit=(10,100),mean=np.random.choice([-5,-4,-3,-2,-1,0,1,2,3,4,5]),p=.5),
                           A.Resize(height=224, width=224),
                           A.Normalize(),
                           ToTensor()])



class Transform():
    def __init__(self,transform):
       self.transform=transform
    def __call__(self,image):
       return self.transform(image=np.array(image))["image"]
   
    
train_dir = 'C:/Users/ProArt/Desktop/DATASET/TRAIN'
# train_dataset = ImageFolder(train_dir, transform=data_transforms) # used for torchvision.transforms to augment images
train_dataset = ImageFolder(train_dir, transform=Transform(albumentations_transforms)) # used for albumentations to augment images


class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
     
        self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = EfficientNet.from_pretrained("efficientnet-b0")

         
        self.resnet.fc = nn.Identity()
        self.efficientnet._fc = nn.Identity()
        
        resnet_output_dim = 2048
        efficientnet_output_dim = 1280
        
        self.fc = nn.Sequential(
            nn.Linear(resnet_output_dim + efficientnet_output_dim, 128),
            nn.ReLU(),
            # nn.Dropout(p=.2),
            nn.Linear(128, num_classes)
            )
        
    def forward(self, x):
         
        resnet_out = self.resnet(x)  # Shape: (batch_size, resnet_channels)
        efficientnet_out = self.efficientnet(x)  # Shape: (batch_size, efficientnet_channels)

        
        combined_features = torch.cat((resnet_out, efficientnet_out), dim=1)  # Shape: (batch_size, combined_channels)
        
        # Classify
        output = self.fc(combined_features)  # Shape: (batch_size, num_classes)
        return output


# model = CombinedModel(num_classes=5)
# x = torch.randn(32, 3, 224, 224)  # Example input
# output = model(x)

model = CombinedModel(num_classes=5)
model.to(device)

def train_model(train_loader, model, criterion, optimizer, scaler, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    itr = 0
    accumulation_steps = 64 // len(train_loader)

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        # loss.backward()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        #Gradient Accumulation
        if (itr + 1 ) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        itr += 1
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return train_loss / len(train_loader), correct / total


def cross_validation(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).to(device)
                
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), correct / total

monte_carlo_iter = 10
scores = {}
for itr in range(monte_carlo_iter):
    kfold = KFold(n_splits = 2, shuffle=True, random_state=itr)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    print(f"----------- Monte Carlo Iteration: {itr + 1} -------------\n")

    # K-fold CV
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        start = time.strftime("%H:%M:%S")
        print(f"----------- Fold: {fold + 1}  --- time: {start} -----------")
    
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)    
    
    
        n_epoch_stop = 10
        best_val_loss = float('inf')
        val_no_improve = 0
        max_epoch_num = 100
        for epoch in range(max_epoch_num):
            train_loss, train_acc = train_model(train_loader, model, criterion, optimizer, scaler, device)
            val_loss, val_acc = cross_validation(val_loader, model, criterion,device)
            scheduler.step(val_loss)
            print(f'Epoch [{epoch + 1}/{max_epoch_num}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_no_improve = 0
                save_path = f'./model-fold-{fold}.pth'
                torch.save(model.state_dict(), save_path)
            else:
                val_no_improve += 1
    
            if val_no_improve >= n_epoch_stop:
                print("Early stopping.")
                break
    
    print('Cross-validation done.')
    key=str("Monte-Carlo-iter-"+str(itr))
    scores[key] = ([train_losses, val_losses,train_accuracies, val_accuracies])

   
# # Visualize training history
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('# of Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Loss History')

# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Training Accuracy')
# plt.plot(val_accuracies, label='Validation Accuracy')
# plt.xlabel('# of Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Accuracy History')

# plt.tight_layout()
# plt.show()

