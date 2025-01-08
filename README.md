# Synthetic Image Classifier

This project utilizes a traditional CNN-to-Linear deep learning model
structure to classify images as either synthetic (A.I. generated) or real.

## Navigating the Project

### /data

The CIFAKE dataset can be found [here](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data). 

Given the size of the dataset, it is not contained within the GitHub repository itself. However, you may download it at the above link, and store it locally in /data.

### /src

All code is in this folder. 

- `eda.ipynb` contains all Exploratory Data Analysis, including example
visualizations and calculating the training dataset's per-channel means and standard deviations for upcoming standardization.

- `dataset.py` contains two critical components: `ImageDataset` and the `get_channel_means_stdevs` function. The former is a custom Dataset to parse
through the aforementioned downloadable dataset's nested file structure. The class
is resilient to new examples, and produces all instances dynamically. The 
latter is a function to calculate the per-channel means and standard deviations
of a particular dataset. This is measured on the training set, and transformed
on the test set (to avoid data leakage). There are many reasons for standardizing the data, including to ensure proper relative feature scaling (for
Stochastic Gradient Descent) and because activation functions' gradients
are often steepest around x = 0.

- `cnn.ipynb` contains the full DL pipeline, from model creation to per-epoch
evaluation of train and test scores. A repeatable visualization is included
at the bottom to see frequent matches between the true and predicted label.

### /results

You may opt to avoid running `get_channel_means_stdevs`, as the results of
this operation have been stored in `channel_training_statistics.pkl` for
immediate use in `cnn.ipynb`.

## Running Locally

This project was completed using Python 3.9.21.

For the complete list of packages, refer to `package_list.txt`. 

For easy installation, utilize `environment.yml`. You can recreate this environment in Conda using `conda env create -f environment.yml`. 