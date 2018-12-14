# Plant Seedlings Classification

Kaggle Competition Project as well as ANLY 590 Final Project. 

## Introduction

This repo is the solution for Kaggle Competition [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification)
as well as the final project of ANLY 590. 

**Task:** Determine the species of a seedling from an image.

More information related to project could be found at [Project Proposal](docs/PROJECT_PROPOSAL.md).

Poster could be found [here](docs/PROJECT_POSTER.pdf).

Final Paper could be found [here](docs/Plant_Seedling_Classification_Final_Paper.pdf).

## Getting Started

### Download Data

You could download data from [website](https://www.kaggle.com/c/plant-seedlings-classification/data) 
or use [Kaggle API](https://github.com/Kaggle/kaggle-api) as following:

```bash
cd input
kaggle competitions download -c plant-seedlings-classification
```

Please download data into [input](input) folder.

### Installing Requirement

```bash
pip install -r requirements.txt
```

### Running Code

```bash
# In Plant_Seedlings_Classification Folder
jupter notebook
```

You could check notebook via Jupter.

## Result

|Method Used|Parameters|F1|
|---|---|---|
|VGG19|20,270,264|0.80|
|6 Layers|3,320,396|0.93306|
|6 Layers + Background Remove|3,320,396|0.95843|
|9 Layers + Fine Tuning|1,775,100|0.97103|
|9 Layers + Background Remove|1,775,100|0.97507|
|9 Layers + ADASYN|1,775,100|0.97799|
|9 Layers + Random Up Sample|1,775,100|0.97744|
|9 Layers + SMOTE|1,775,100|0.97832|
|9 Layers + ADASYN + Data Augmentation|1,775,100|0.97902|
|9 Layers + ADASYN + Data Augmentation + Snapshot Ensemble|1,775,100|0.98740|
|CNN DenseNet 121 + Data Augmentation + Background Remove|8,062,504|0.94710|
|CNN DenseNet 121 + Data Augmentation + GAN + Background Remove|8,062,504|0.97984|
|State of Art on Kaggle[8]|54,521,176|0.99496|

## Author

* [Yi Ding](https://github.com/dy11) [yd137@georgetown.edu](mailto:yd137@georgetown.edu)
* [Yu Xiao](https://github.com/troyxiao) [yx151@georgetown.edu](mailto:yx151@georgetown.edu)
* [Zhuoran Wu](https://github.com/WuZhuoran) [zw118@georgetown.edu](mailto:zw118@georgetown.edu)

## Reference

[1] Thomas Mosgaard Giselsson and (2017). A Public Image Database
for Benchmark of Plant Seedling Classification. CoRR,
abs/1711.05458.

[2] Giselsson, Thomas Mosgaard, Dyrmann, Mads, Jorgensen, Rasmus
Nyholm, Jensen, Peter Kryger & Midtiby, Henrik Skov (2017). A
Public Image Database for Benchmark of Plant Seedling
Classification Algorithms.

[3] J. Schwiegerling, Field Guide to Visual and Ophthalmic Optics, SPIE
Press, Bellingham, WA (2004).

[4] Haibo He, Yang Bai, E. A. Garcia and Shutao Li, "ADASYN:
Adaptive synthetic sampling approach for imbalanced learning," 2008
IEEE International Joint Conference on Neural Networks (IEEE
World Congress on Computational Intelligence), Hong Kong, 2008,
pp. 1322-1328. doi: 10.1109/IJCNN.2008.4633969

[5] Chawla, N. V. et al. “SMOTE: Synthetic Minority Over-Sampling
Technique.” Journal of Artificial Intelligence Research 16 (2002):
321–357. Crossref. Web.

[6] Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger,
K. Q. (2017). Snapshot ensembles: Train 1, get M for free. arXiv
preprint arXiv:1704.00109 .

[7] Guillaume Lemaitre, Fernando Nogueira & Christos K. Aridas
(2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of
Imbalanced Datasets in Machine Learning. Journal of Machine
Learning Research, 18, 1-5.

[8] Kumar Shridhar., Kaggle #1 Winning Approach for Image
Classification Challenge. Web Blog Post, Neural Space, June 20, 2018.

[9] Karl Pearson, 1901 K. Pearson, "On lines and planes of closest fit to
systems of points in space", The London, Edinburgh and Dublin
Philosophical Magazine and Journal of Science, Sixth Series, 2, pp.559-572 (1901) (1857-1936)

[10] L.J.P. van der Maaten and G.E. Hinton. Visualizing Data Using
t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605,2008.

[11] Weiss, Karl, et al. “A Survey of Transfer Learning.” Journal of Big
Data , vol. 3, no. 1, 2016.

[12] Hinterstoisser, Stefan, et al. “On Pre-Trained Image Features and
Synthetic Images for Deep Learning.” 2017.

[13] Wu, Cinna, Mark Tygert, and Yann LeCun. "Hierarchical loss for
classification." arXiv preprint arXiv:1709.01062 (2017).
5 Radford A, Metz L, Chintala S. Unsupervised representation
learning with deep convolutional generative adversarial networks[J].
arXiv preprint arXiv:1511.06434, 2015.

[14] Gao F, Yang Y, Wang J, et al. A deep convolutional generative
adversarial networks (DCGANs)-based semi-supervised method for
object recognition in synthetic aperture radar (SAR) images[J]. Remote Sensing, 2018, 10(6).