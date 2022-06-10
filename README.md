##About main programs
The four '.py' files are the main programs we construct to train and test accuracies of our models with different structures.You can also obtain the plots of training loss together with test loss. The names of the files indicate the structures we use to build our models:

In 'prototype.py', you can train and validate the model with the structure we called 'prototype' in our final report. 

In 'batchnorm.py', you can train and validate the model with the structure we called 'prototype + batchnorm' in our final report. 

In 'ReLU.py', you can train and validate the model with the structure we called 'prototype + batchnorm + ReLU' in our final report. 

In 'VGG11.py', you can train and validate the model with the structure we called 'VGG11' in our final report. 
##About dataset
The original version of the dataset we used is from : 
https://www.kaggle.com/datasets/moltean/fruits. 

The set contains 90380 images of 131 fruits and vegetables and is constantly updated with images of new fruits and vegetables. In our project, we selected 17629 images of 35 fruits as the data set of our project. which you can find in folder 'dataset'. 
##How to run our code and reproduce our results
To reproduce the results we presented in the report, you should follow the instructions below:
###Preprocess the dataset:
1.Download the data sets from the link:
https://drive.google.com/drive/folders/1pfWUf1j4nBMo15GT66Rggy7HVOq73V_u?usp=sharing
2.Create a new folder under the same directory of this 'README.md'.
3.Unzip 'training.zip' and 'test.zip' in the folder 'dataset'.
4.Run the code named as 'data.py' to obtain the 4-channel (RGB + grey scale value) tensor of images. You can follow the comments in code to preprocess both training dataset and test dataset by modify the code slightly.

5.To augment the training dataset, you should run 'data_process.py' after you obtaining the 4-channel tensor('training_data_before_per.pt' and 'training_label_before_per.pt').

####Note: 
Preprocessing may cost a lot of time. To save your time, you can download the dataset we have already preprocessed. 

Link:https://drive.google.com/file/d/18WZUNT5gOkbtOQt0mKDUiSJrJ4p6fSx0/view?usp=sharing

######Just unzip the file under the same directory of this 'README.md'.

###Training and test accuracies of models:
1.To train the different models mentioned in our report with different structures, you should run the corresponding program.

2.To reproduce our results, you should modify the hyperperameters in the dictionary named as 'trainer_info'.


######Please refer to the comments in every code for some more detailed instruction and program description.
