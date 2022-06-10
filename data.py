import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

folder='dataset/test'
#This code is used to preprocess test dataset. If you want to preprocess training dataset,
#please change the folder as 'dataset/training'
i=0
j=0
test_data = np.empty([1,100,100,4])
test_label=np.empty([1,35])
for foldername in tqdm([fo for fo in os.listdir(folder) if not fo.startswith('.')]):
    i=0
    for filename in [f for f in os.listdir(folder+'/'+foldername) if not f.startswith('.')]:
        img = cv2.imread(os.path.join(folder+'/'+foldername,filename))
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = img2.reshape(100, 100, 1)
        img = np.concatenate((img1, img2), axis=2)
        img = img.reshape(1, 100, 100, 4)
        test_data=np.concatenate((test_data,img),axis=0)
        i=i+1
    test_label_tem=np.zeros(shape=(i,35))
    test_label_tem[:,j]=1
    test_label=np.concatenate((test_label,test_label_tem),axis=0)
    j = j + 1

test_data=np.delete(test_data,0,axis=0)
test_label=np.delete(test_label,0,axis=0)
test_data=torch.tensor(test_data)
test_label=torch.tensor(test_label)
test_data=test_data.permute(0,3,1,2)

if folder=='dataset/test':
    torch.save(test_data, 'test_data.pt')
    torch.save(test_label, 'test_label.pt')
elif folder=='dataset/training':
    torch.save(test_data, 'training_data_before_per.pt')
    torch.save(test_label, 'training_label_before_per.pt')
    pass

