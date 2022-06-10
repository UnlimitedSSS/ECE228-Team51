import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

#This is a program to augment training data by flipping training images.
training_data=torch.load('training_data_before_per.pt')
training_label=torch.load('training_label_before_per.pt')

training_data_rotated=np.zeros(shape=(17628,4,100,100))
for k in tqdm(range(17628)):
    for m in range(3):
        for i in range(100):
            for j in range(100):
                training_data_rotated[k,m,-j+99,i]=training_data[k,m,i,j]

training_data_rotated=torch.tensor(training_data)
training_data_rotated=torch.cat((training_data,training_data_rotated),0)
train_label_rotated=torch.cat((training_label,training_label),0)

rng = np.random.default_rng()
numbers = rng.choice(35256, size=35256, replace=False)

training_data_final = training_data_rotated[numbers]
train_label_final = train_label_rotated[numbers]

torch.save(training_data_final,'training_data.pt')
torch.save(train_label_final,'training_label.pt')