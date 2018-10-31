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

## Assessment Metrics

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
