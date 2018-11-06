# Project Proposal

## Introduction

For several decades, researchers have worked on systems aimed at performing site-specific weed control.
Although some systems are commercially available, 
a true commercial breakthrough of such systems is still to come despite the construction
of several prototypes and case studies showing promising results.

In this project, we will work to build a classification model for 
a [database](https://vision.eng.au.dk/plant-seedlings-dataset) 
of images of approximately 960 unique plants belonging to 12 species at several growth stages.
It is also a Kaggle [competition](https://www.kaggle.com/c/plant-seedlings-classification).

The code will be available online at [https://github.com/WuZhuoran/Plant_Seedlings_Classification].

## Team

* [Yi Ding]()
* [Yu Xiao]()
* [Zhuoran Wu](https://github.com/WuZhuoran) [zw118@georgetown.edu](mailto:zw118@georgetown.edu)

## Project's Goal and Objectives

## Data

The Plant Seedlings Dataset contains images of approximately 960 unique plants belonging to 12 species at several growth stages. It comprises annotated RGB images with a physical resolution of roughly 10 pixels per mm.The dataset comes from Aarhus University Flakkebjerg Research station in a collaboration between University of Southern Denmark and Aarhus University (https://vision.eng.au.dk/plant-seedlings-dataset/). This is the only data source that will be used in our project.  The dataset is publicly available and used in a Kaggle challenge. The dataset is collected by professional researchers specifically for the plant classification task and is a reliable data source. The goal of the competition is to create a classifier capable of determining a plant's species from a image. The list of species are: Black-grass, Charlock, Cleavers, Common Chickweed, Common Wheat, Fat Hen, Loose Silky-bent,  Maize, Scentless Mayweed, Shepherds Purse, Small-flowered Cranesbill, Sugar beet. 
The size of each species is shown in the pie chart:
<img width="721" alt="speciessize" src="https://user-images.githubusercontent.com/7198810/48086786-ae14a780-e1cb-11e8-877d-9b3b02157062.png">

Following are sample images for each species:

<img width="1381" alt="screen shot 2018-11-06 at 1 59 51 pm" src="https://user-images.githubusercontent.com/7198810/48087093-66dae680-e1cc-11e8-81e8-96d878418bf2.png">


This is a supervised learning task. There are two sets of images (training and testing) provided for study. A training set of images is organized by plant species in 12 seperate folders. Each images has a unique id that can be easily linked to its plant species. A testing set of images is just a mix of 12 plant species. 
There are a few limitation of this dataset. First,  the size of training set is relatively small. On average, there are about 400 images in each class. Small dataset could result in overfitting. It is hard to collect more data since the background of the Plant Seedlings Dataset are unique. Most images have bar code, pebblestone, metal device and wall. Images from different source are very likely to have other background and disturb the model learning. 


<img width="772" alt="screen shot 2018-11-06 at 2 11 12 pm" src="https://user-images.githubusercontent.com/7198810/48087651-e5845380-e1cd-11e8-8cc4-33424adf28d9.png">

Second, the dataset is imbalanced. For example, there are 654 Loose Silky-bent images but only 221 Common Wheat images. In this case, the model could be inclined to predicting larger class. Third, images in this dataset come in different size. Small images is about 75*75 pixels while large image can be 3991*3457 pixels. Even if we could resize all images to 224*224 pixels, which is a common input size for pre-trained model, small images would be in low quality (visually blurry) since they are enlarged. 



## Assessment Metrics

We will select categorical cross entropy as our loss function since the categorical cross entropy is preferred for mutually-exclusive multi-class classification task (where each example belongs to a single class) compared to other metrics. 

We will use provided testing set as the baseline dataset for models evaluation. The baseline model of our project is multi-class Logistic regression. We will choose micro-averaged F1 score as the major evaluation matrix since it is selected in the Kaggle competition, and it is easier to compare the performance of our model with previous works. In addition, F1 score could balance the precision and recall and yield a more realistic indication of model performance. We would also use confusion matrix to visualize the prediction results.

Besides baseline models, we plan to experiment with models such as simple Neural Network model, CNN models (pre-trained models) and ensemble model with XGBoost to see their performances. The state of the art F1 score is achieved by a customized 10 layer CNN models. Pre-trained model, Xception, also yields a similar state of art result. 



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
   
3. Convolutional Neural Network with Pre-trained Model     
   We will try different Pre-trained Model. Those model will either be used for extracting features or
   for classifiction.
   
4. Convolutional Neural Network with regularization, optimizer and other method    
   We will introduce some regularization method in NN model which includes but noe limited as:
   * Regularization
   * Drop Out
   * Optimizer
   We will compare the results with other results.
   
5. Ensemble Method with XGBoost.     
   We will use ensemble method for our different models.
   
6. Deep Learning Method + Traditional Method (SVM)    
   We will try to find a way to combine both traditional method and deep learning methods. 
   
7. Data Augmentation     
   Data pre-processing and data augmentation will also be used in the task.


### Possible limitations of approach

Currently we mainly consider `Convolutional Neural Network` as our network structure. 
Therefore we might face those problems:

* Hyperparamter tuning is non-trivial
* Need a large dataset
* The scale of a net's weights (and of the weight updates) is very important for performance. When the features are of the same type (pixels, word counts, etc), this is not a problem. However, when the features are heterogeneous--like in many Kaggle datasets--your weights and updates will all be on different scales (so you need to standardize your inputs in some way).
* cost effective

### Tools & API 

The details of python packages are listed in the [requirements](requirements.txt) file.

We will mainly use [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) as our Deep Learning Framework. 
Also [PyTorch](https://pytorch.org/) will be used in some approach. 

We will use [sklearn](http://scikit-learn.org/) and [skimage](https://scikit-image.org/) for data processing and traditional method.

### Training & Test Platform

We will train and test our task both on local and cloud machine.

* Cloud Machine

Clouc Machine we use `AWS`, and `Kaggle`.

* Local Machine

### Limitations of platform

For Cloud Machine, the platform is hard to control and has less flexibility. 
Also the cost will be large if we only use enterprise cloud machine.

For Local Machine, it might be slow and short of memory because local computer do not always have a good CPU or RAM.

## Reference

[1] Thomas Mosgaard Giselsson, Rasmus Nyholm Jørgensen, Peter Kryger Jensen, Mads Dyrmann: "A Public Image Database for Benchmark of Plant Seedling Classification Algorithms", 2017; [http://arxiv.org/abs/1711.05458 arXiv:1711.05458].

[2] Mads Dyrmann, Henrik Karstoft, Henrik Skov Midtiby: "Plant species classification using deep convolutional neural network", 2016; [https://www-sciencedirect-com.proxy.library.georgetown.edu/science/article/pii/S1537511016301465]
