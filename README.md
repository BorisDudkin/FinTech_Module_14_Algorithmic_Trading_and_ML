# Algorithmic Trading and Machine Learning Models.

### Algorithmic Trading powered by Machine Learning models enables investors to automatically trade assets in highly dynamic and volatile environments. Machine learning systems can evaluate multiple factors that influence prices of financial instruments and efficiently generate accurate trading decisions.<br/>This application takes you through the process of implementing the Machine Learning driven Algorithmic Trading strategies.

---

![ecommerce](Images/algo-trading.jpg)

---

## Table of contents

1. [Technologies](#technologies)
2. [Installation Guide](#installation-guide)
3. [Usage](#usage)
4. [Contributors](#contributors)
5. [License](#license)

---

## Technologies

`Python 3.9`

`Jupyter lab`

_Prerequisites_

1. `Pandas` is a Python package that provides fast, flexible, and expressive data structures designed to make working with large sets of data easy and intuitive.

   - [pandas](https://github.com/pandas-dev/pandas) - for the documentation, installation guide and dependencies.

2. `Scikit-learn` is a simple and efficient tools for predictive data analysis. It is built on NumPy, SciPy, and matplotlib.

   - [scikit-learn ](https://scikit-learn.org/stable/) - for information on the library, its features and installation instructions.<br/>

---

## Installation Guide

Jupyter lab is a preferred software to work with Risk Return Analysis application.<br/> Jupyter lab is a part of the **[anaconda](https://www.anaconda.com/)** distribution package and therefore it is recommended to download **anaconda** first.<br/> Once dowloaded, run the following command in your terminal to lauch Jupyter lab:

```python
jupyter lab
```

Before using the application first install the following dependencies by using your terminal:

To install pandas run:

```python
#  PuPi
pip install pandas
```

```python
# or conda
conda install pandas
```

To install scikit library, in Terminal run:

```python
# PuPi
pip install -U scikit-learn
```

---

## Usage

> Application summary<br/>

Machine Learning Models and Venture Capital application assist in creating and evaluating a neural network model that predicts whether applicants will be successful if funded by a venture capital firm. The tool takes a user through three steps involving identifying, evaluating and optimizing the neural network models:<br/>

- Processing of the data for a neural network model.
- Using the model-fit-predict pattern to compile and evaluate a binary classification model.
- Optimizing the model.

**Processing of the data for a neural network model:**<br/>

- In this part of the applcation we load the dataset and chek it for the categorical values: <br/>
  ![categotrical](Images/CategoricalPNG.PNG)<br/>
- Once identified, those values are transformed into numerical values with OneHotEncoder and concatenated with other numerical data from the original dataset.
- The data is split into the features and the target and into the training and testing sets. The features are then scaled with the StandardScaler.

**Compiling and Evaluating the Binary Classification Model Using a Neural Network:**<br/>

- We start by creating a deep neural network with a two-layer deep neural network model that uses the relu activation function for both layers and the output activation function sigmoid appropriate for the classifier models. Below is the summary of the model we created:<br/>
  ![m1_summary](Images/m1_summary.PNG)<br/>
- We compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric. 50 epochs are used at this stage.<br/>
- Next, we evaluate the model by using the scaled test data and the test target values <br/>
  ![m1_reval](Images/M1_eval.PNG)<br/>
- We finalize this part by saving and exposrting the model we created to our Resources folder. <br/>

**Optimizing the neural network model:**<br/>

_Model 1_<br/>

- We first optimize the model by adjusting the dataset: two features are identified as imbalanced and removed: STATUS and SPECIAL_CONSIDERATIONS. Below are the details of the imbalances for those two features:

  - STATUS:<br/>
    ![status](Images/status.PNG)<br/>
  - SPECIAL_CONSIDERATIONS:<br/>
    ![status](Images/Considerations.PNG)<br/>

- We compile the model using the rest of the features and characteristics of the original model:

  ![m2_summary](Images/m2_summary.PNG)<br/>

- The evaluation produces the following results:<br/>
  ![m2_reval](Images/M2_eval.PNG)<br/>

_Model 1a_

- We optimize this model further by increasing the number of epochs to 100 from 50 with the evaluation results demostrated below:<br/>
  ![m2a_reval](Images/M2a_eval.PNG)<br/>

_Model 2_

- This optimization involves reverting to the original data set and modifying the original model parameters as follows:<br/>
  - Added one hidden layer;
  - Hidden layers activation function changed to Leaky ReLU;
  - Epochs increased from 50 to 100.<br/>
- Below is the summary of this model:<br/>
  ![m3_summary](Images/m3_summary.PNG)<br/>

- The evaluation of this model produces the following results:<br/>
  ![m3_reval](Images/M3_100_eval.PNG)<br/>

**Summary**:<br/>
The loss and accuracy parametes of all our models are compared below:<br/>
![summary](Images/Summary.PNG)<br/>
By removing two imbalanced features we managed to increase the model's loss but the accuracy parameter slightly deteriorted.<br/> Interestingly, increasing the number of epochs did not imporve the optimized model's performance.<br/>
When reverting back to the original dataset and changing the model's parameters we improved the loss metrics but the accuracy decreased comparing to the original model. However, in terms of the loss model M2 is is still underperforming model M1 but overperforming M1 on accuracy.<br/>
Overall, we did not succeed to significantly improve the original model's performance. In this case the subject matter expert's opinion will be decisive in the ultimate model selection.<br/>

> Getting started<br/>

- To use Portoflio Optimizer first clone the repository to your PC.<br/>
- Open `Jupyter lab` as per the instructions in the [Installation Guide](#installation-guide) to run the application.<br/>

---

## Contributors

Contact Details:

Boris Dudkin:

- [Email](boris.dudkin@gmail.com)
- [LinkedIn](www.linkedin.com/in/Boris-Dudkin)

---

## License

MIT

---
