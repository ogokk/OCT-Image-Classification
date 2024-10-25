# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:18:03 2024

@author: ozangokkan
"""
import torch
from train import CombinedModel
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score,matthews_corrcoef, balanced_accuracy_score, recall_score
preds_classes = []
f1_scores_per_batch = []
mcc_scores_per_batch = []
bas_per_batch = []
recall_per_batch = []

# Data preprocessing
test_data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dir = 'C:/Users/ProArt/Desktop/DATASET/TEST'

test_dataset = ImageFolder(test_dir, transform=test_data_transforms)
# # print(test_dataset.imgs) # to check label values

monte_carlo_iter = 100
shuffle_dataset = True
batch_size = 8
torch.cuda.empty_cache()
preds_classes = []
f1_scores_per_batch = []
mcc_scores_per_batch = []
bas_per_batch = []
with torch.no_grad():
    model = CombinedModel(num_classes=5)  # Use the same num_classes as in training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    path = 'C:/Users/ProArt/Desktop/ozan/model-fold-4.pth'
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode

for i in range(monte_carlo_iter):
    random_seed = i
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(1 * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    test_indices = indices
    
    # Creating data samplers and loaders:
    test_sampler = SubsetRandomSampler(test_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                                sampler=test_sampler)
    
    for i, (inputs, classes) in enumerate(tqdm(test_loader)):
        inputs, classes = next(iter(test_loader))
        inputs=inputs.to(device)
        outputs=model(inputs)
        # print(outputs)
        _, preds = torch.max(outputs, 1)
        preds=preds.cpu().numpy()
        classes=classes.numpy()
        f1_scores_per_batch.append(f1_score(classes, preds, average='macro'))
        mcc_scores_per_batch.append(matthews_corrcoef(classes, preds))
        bas_per_batch.append(balanced_accuracy_score(classes, preds))
        recall_per_batch.append(recall_score(classes, preds, average="macro"))
        preds_classes.append([preds, classes])

print("\nF1 macro score is :%{:.2f}".format(100*np.mean(f1_scores_per_batch)))
print("\nMCC score is :%{:.2f}".format(100*np.mean(mcc_scores_per_batch)))
print("\nBalanced Accuracy score is :%{:.2f}".format(100*np.mean(bas_per_batch)))
print("\nSensitivity (Recall) score is :%{:.2f}".format(100*np.mean(recall_per_batch)))






# #-------------------------SINGLE IMAGE AND BATCH TESTING---------------------
# #Single batch prediction
# inputs, classes = next(iter(test_loader))
# inputs=inputs.to(device)
 
# outputs=model(inputs)
# _, preds = torch.max(outputs, 1)
# preds=preds.cpu().numpy()
# classes=classes.numpy()
# print(preds)
# print(classes)




#--------------------------------------------------------------------------------------
# #Single Image prediction
# from PIL import Image
# import cv2

# def pre_image(image_path,model):
#    img = Image.open(image_path)
#    img = img.convert('RGB')
#    mean = [0.485, 0.456, 0.406] 
#    std = [0.229, 0.224, 0.225]
#    transform_norm = transforms.Compose([transforms.ToTensor(), 
#    transforms.Resize((224,224)),transforms.Normalize(mean, std)])
#    # get normalized image
#    img_normalized = transform_norm(img).float()
#    img_normalized = img_normalized.unsqueeze_(0)
#    img_normalized = img_normalized.to(device)
#    # print(img_normalized.shape)
#    with torch.no_grad():
#       model.eval()  
#       output =model(img_normalized)
#      # print(output)
#       index = output.data.cpu().numpy().argmax()
#       classes = ["glioma","meningioma","normal","pituitary"]
#       class_name = classes[index]
#       return class_name

# predict_class = pre_image('C:/Users/ozangokkan/Desktop/brainmri/dataset/Testing/pituitary/Te-pi_0010.jpg',model)
# print(predict_class)


