# Project Proposal

## Introduction

For several decades, researchers have been worked on systems that aim to perform site-specific weed control.
Although some systems are commercially available, 
a true commercial breakthrough of such systems is still to come despite the construction
of several prototypes and case studies showing promising results.

In this project, we will build serveral classification models on 
a [database](https://vision.eng.au.dk/plant-seedlings-dataset) 
of images of approximately 960 unique plants belonging to 12 species at several growth stages.
This task is also a Kaggle [competition](https://www.kaggle.com/c/plant-seedlings-classification).

The code will be available online at [https://github.com/WuZhuoran/Plant_Seedlings_Classification].

## Team

* [Yi Ding](https://github.com/dy11) [yd137@georgetown.edu](mailto:yd137@georgetown.edu)
* [Yu Xiao](https://github.com/troyxiao) [yx151@georgetown.edu](mailto:yx151@georgetown.edu)
* [Zhuoran Wu](https://github.com/WuZhuoran) [zw118@georgetown.edu](mailto:zw118@georgetown.edu)

## Project's Goal and Objectives

We choose to participate in the kaggle competition Plant Seedlings Classification as our final project.
Plant Seedlings Classification is a typical supervised, multiclass classification problem. We aim to identify the different weed from a crop seedling image.

* DataSet:

The dataset has 12 different plaint classes. The image number of each class is as follow.

| image number of each class | 
| :------| 
| Black-grass	263 | 
| Charlock	390 | 
| Cleavers	287 | 
| Maize		221 | 
| Common Chickweed	611 | 
| Common wheat	221 | 
| Fat Hen	475 | 
| Loose Silky-bent	654 | 
| Scentless Mayweed 516 | 
| Shepherds Purse	231 | 
| Small-flowered Cranesbill	496 | 
| Sugar beet	385 | 

The expected end result of this task for each test query is a probability distribution of the classification.

We take the pre-trained model resnet50 as our benchmark. We will freeze layers except the fully connected layer and set Learning rate = 0.001 and training 20 epochs.

### Performance on val dataset：

             precision    recall  f1-score   support

          0       0.60      0.94      0.73        50
          1       1.00      0.96      0.98        50
          2       0.94      0.94      0.94        50
          3       0.74      0.92      0.82        50
          4       0.91      0.82      0.86        50
          5       1.00      0.94      0.97        50
          6       0.82      0.56      0.67        50
          7       0.91      0.98      0.94        50
          8       0.87      0.96      0.91        50
          9       0.97      0.62      0.76        50
         10       1.00      0.94      0.97        50
         11       0.92      0.88      0.90        50


- Epcch 19: loss=0.62 acc=0.87

After that, we will use some different evaluation methods and try a couple of different network models.

We will also propose serval unique idea to tacle this problem better.

*	Generate mask for seeds by semantic segmentation:

We will manually label some of the images, combine the data from kaggle and the tutorial from opencv and apply semi-supervised learning.

*	Use GAN to augment data:

The image size distribution of dataset is:

[0, 64) 1.26%

[64, 96)  7.73%

[96, 128) 11.28%

[128, 160)  12.06%

[160, 192)  10.11%

[192, 224)  3.89%

[224, 256)  3.12%

[256, 512)  25.96%

[512, 1024) 19.96%

[1024, 2048)  4.38%

[2048, 4096]  0.25%

We can see that there are almost half of the image are smaller than 225*225 pixels. Resizing the image smaller than 224*224 pixels to a larger size may not be a good practice. We will try to use GAN instead of resizing for data augmentation. 

*	Hierarchical loss:

Since some of the crops are really similar, we can calculate the confusion matrix for each pair of class. We will consider about combining relatively large confusion as a superclass, and adding an auxiliary branch on our network for fine-grain classification in the superclass, e.g. the hierarchical softmax from word2vec. 




## Data

The Plant Seedlings Dataset contains images of approximately 960 unique plants belonging to 12 species at several growth stages. It comprises annotated RGB images with a physical resolution of roughly 10 pixels per mm. The dataset comes from Aarhus University Flakkebjerg Research station in a collaboration between University of Southern Denmark and Aarhus University (https://vision.eng.au.dk/plant-seedlings-dataset/). This is the only data source that will be used in our project.  The dataset is publicly available and used in a Kaggle challenge. The dataset is collected by professional researchers specifically for the plant classification task and is a reliable data source. The goal of the competition is to create a classifier capable of determining a plant's species from a image. The list of species are: Black-grass, Charlock, Cleavers, Common Chickweed, Common Wheat, Fat Hen, Loose Silky-bent,  Maize, Scentless Mayweed, Shepherds Purse, Small-flowered Cranesbill, Sugar beet. 
The size of each species is shown in the pie chart:
<img width="721" alt="speciessize" src="https://user-images.githubusercontent.com/7198810/48086786-ae14a780-e1cb-11e8-877d-9b3b02157062.png">

Following are sample images for each species:

<img width="1381" alt="screen shot 2018-11-06 at 1 59 51 pm" src="https://user-images.githubusercontent.com/7198810/48087093-66dae680-e1cc-11e8-81e8-96d878418bf2.png">


This is a supervised learning task. The input feature vector is images and the output is probability distribution of the classification. There are two sets of images (training and testing) provided for study. A training set of images is organized by plant species in 12 seperate folders. Each images has a unique id that can be easily linked to its plant species. A testing set of images is just a mix of 12 plant species. 
There are a few limitation of this dataset. First,  the size of training set is relatively small. On average, there are about 400 images in each class. Small dataset could result in overfitting. It is hard to collect more data since the background of the Plant Seedlings Dataset are unique. Most images have bar code, pebblestone, metal device and wall. Images from different source are very likely to have other background and disturb the model learning. 


<img width="772" alt="screen shot 2018-11-06 at 2 11 12 pm" src="https://user-images.githubusercontent.com/7198810/48087651-e5845380-e1cd-11e8-8cc4-33424adf28d9.png">

Second, the dataset is imbalanced. For example, there are 654 Loose Silky-bent images but only 221 Common Wheat images. In this case, the model could be inclined to predicting larger class. Third, images in this dataset come in different size. Small images is about 75*75 pixels while large image can be 3991*3457 pixels. Even if we could resize all images to 224*224 pixels, which is a common input size for pre-trained model, small images would be in low quality (visually blurry) since they are enlarged. 



## Assessment Metrics

We will select categorical cross entropy as our loss function since the categorical cross entropy is preferred for mutually-exclusive multi-class classification task (where each example belongs to a single class) compared to other metrics. 

We will use the whole dataset as the baseline dataset for models evaluation because the Plant Seeding dataset is pretty small. The baseline model of our project is CNN with ResNet50 and multi-class Logistic regression. We will choose micro-averaged F1 score as the major evaluation matrix since it is selected in the Kaggle competition, and it is easier to compare the performance of our model with previous works. In addition, F1 score could balance the precision and recall and yield a more realistic indication of model performance. We would also use confusion matrix to visualize the prediction results.

Besides baseline models, we plan to experiment with models such as simple Neural Network model, CNN models (pre-trained models) and ensemble model with XGBoost to see their performances. The state of the art F1 score is achieved by a DenseNet 161 model (F-1 score 98.236). Pre-trained model, Xception, also yields a similar state of art result. 


## Approach

According to the paper [1], we found that the paper present a segmentation method by Naive Bayes.
However, recently deep learning method is widely used in machine learning task. 
This time, our team will use Deep Learning Method to solve the classification problem.
And we will also try some traditional methods in order to compare the result.

In paper [2], the authors present method about plant identification with Deep Learning Method.  

### Approach List

Here is our Approach List:

1. Logistic Regression   
   We want to use multi-class Logistic regression to build a quick baseline.
   
2. Neural Network Method    
   We will build some simple neural network to test if Deep Learning could do great in this task.
   
3. Convolutional Neural Network(CNN) with Pre-trained Model 
   CNN is very good at image classification. There are many pre-trained Model in libraries such as Keras, in this project,
   we will experiment with different Pre-trained Model. Models will either be used for features selection or classifiction.
   There are serveral benefits of applying pretrained models. For example, pre-trained models can save training time and it  
   does not require a lot of data to train a Neural Network from scratch. Pre-trained models usually yield a better results 
   than stacking a CNN model manually. 
   
4. Convolutional Neural Network with regularization, optimizer and other method    
   We will introduce some regularization method in NN model which includes but noe limited as:
   * Regularization
   * Drop Out
   * Optimizer
   We will compare the results with other results.
   
5. Ensemble Method with XGBoost.     
   We will use ensemble method for our different models. Ensemble methods can potentially combine the strengh of individul models. 
   
6. Deep Learning Method + Traditional Method (SVM)    
   We will try to combine both traditional method and deep learning methods. 
   
7. Data Augmentation     
   Data pre-processing and data augmentation will be used in the task.


### Possible limitations of approach

We mainly consider `Convolutional Neural Network` as our network structure. 
And we might run into the following challenges:

* Hyperparamter tuning is non-trivial
* Need a large dataset
* The scale of a net's weights (and of the weight updates) is very important for performance. When the features are of the same type (pixels, word counts, etc), this is not a problem. However, when the features are heterogeneous--like in many Kaggle datasets--your weights and updates will all be on different scales (so you need to standardize your inputs in some way).
* cost effective

### Tools & API 

Details of python packages are listed in the [requirements](requirements.txt) file.

We will mainly use [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) as our Deep Learning Framework. 
Also [PyTorch](https://pytorch.org/) will be used in some approach. 

We will use [sklearn](http://scikit-learn.org/) and [skimage](https://scikit-image.org/) for data processing and traditional method.

### Training & Test Platform

We will train and test our task both on cloud machine and local machine.

* Cloud Machine

We will mainly use `AWS`, and `Kaggle`.

* Local Machine

### Limitations of platform

For Cloud Machine, the platform is hard to control and has less flexibility. 
Also the cost will be large if we only use enterprise cloud machine.

For Local Machine, it might be slow and short of memory because local computer do not always have a good CPU or RAM.

## Reference

[1] Thomas Mosgaard Giselsson, Rasmus Nyholm Jørgensen, Peter Kryger Jensen, Mads Dyrmann: "A Public Image Database for Benchmark of Plant Seedling Classification Algorithms", 2017; [http://arxiv.org/abs/1711.05458 arXiv:1711.05458].

[2] Mads Dyrmann, Henrik Karstoft, Henrik Skov Midtiby: "Plant species classification using deep convolutional neural network", 2016; [https://www-sciencedirect-com.proxy.library.georgetown.edu/science/article/pii/S1537511016301465]
