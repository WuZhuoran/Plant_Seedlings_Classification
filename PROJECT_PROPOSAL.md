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
The size of each specie is shown in the pie chart:

<img width="721" alt="speciessize" src="https://user-images.githubusercontent.com/7198810/48086786-ae14a780-e1cb-11e8-877d-9b3b02157062.png">



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


## Reference

[1] Thomas Mosgaard Giselsson, Rasmus Nyholm Jørgensen, Peter Kryger Jensen, Mads Dyrmann: "A Public Image Database for Benchmark of Plant Seedling Classification Algorithms", 2017; [http://arxiv.org/abs/1711.05458 arXiv:1711.05458].

[2] Mads Dyrmann, Henrik Karstoft, Henrik Skov Midtiby: "Plant species classification using deep convolutional neural network", 2016; [https://www-sciencedirect-com.proxy.library.georgetown.edu/science/article/pii/S1537511016301465]
