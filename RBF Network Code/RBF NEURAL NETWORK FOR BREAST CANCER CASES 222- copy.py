#!/usr/bin/env python
# coding: utf-8

# In[1]:


### BREAST CANCER CASES ###
###### SVM WITH RBF KERNEL CODE IN JUPYTER NOTEBOOK #####


# In[2]:


## Modules required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import keras
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# In[3]:


# Code
BC = (pd.read_excel('cancer.xlsx'))


# In[4]:


BC.head()


# In[5]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

#Import numpy#
import numpy as np


# In[6]:


y = BC.PatStatus
x = BC.drop(['PatStatus'], axis = 1)


# In[7]:


#Split the data into train and test sets #
#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=123)


## Scaling the data
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(x_train)
x_test_minmax = min_max_scaler.fit_transform(x_test)


# In[8]:


x_train = x_train_minmax
x_test = x_test_minmax


# In[9]:


x_train.shape


# In[10]:


x_test.shape


# In[11]:


## Fitting the model
## Models required
from keras.preprocessing import image 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
from keras.utils import layer_utils, np_utils 
from sklearn.metrics import confusion_matrix, classification_report 
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice, uniform
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.svm import SVC

# SVM classifier with Gaussian RBF kernel
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(x_train,y_train)


# In[12]:


# predict with splitted test data
y_pred = classifier.predict(x_test)


# In[13]:


##Fitting the neural network model using training dataset
tns_probs=[0 for _ in range(len(y_test))]


# In[14]:


## CONFUSION MATRIX FOR BOTH SEX DATA
test_cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(test_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, test_cm[i, j], ha= 'center', va= 'center', color= 'red')
plt.show()


# In[15]:


## Error for the prediction for test dataset outcomes
test_error = (test_cm[0,1] + test_cm[1,0])/np.sum(test_cm)
print(test_error)


# In[16]:


## Accuracy of prediction
1-test_error


# In[17]:


## Sensitivity Analysis 
test_sens = test_cm[1, 1]/(test_cm[1, 1] + test_cm[0, 1])
print(test_sens)


# In[18]:


## Specificity Analysis
test_spec = test_cm[0, 0]/(test_cm[0, 0]+test_cm[1, 0])
print(test_spec)


# In[19]:


## PPV Analysis
test_npv = test_cm[1, 1]/(test_cm[1, 1] + test_cm[1, 0])
print(test_npv)


# In[20]:


## NPV Analysis
test_npv = test_cm[0, 0]/(test_cm[0, 0]+test_cm[0, 1])
print(test_npv)


# In[21]:


## The AUC Score
test_auc = roc_auc_score(y_test, tns_probs)
y_pred_auc = np.round(roc_auc_score(y_test, y_pred), decimals = 2)


# In[22]:


## calculate ROC Curves
test_fpr, test_tpr, _ = roc_curve(y_test, tns_probs)
y_pred_fpr, y_pred_tpr, _ = roc_curve(y_test, y_pred)


# In[23]:


## Plot Curve for the model
import numpy as np
import matplotlib.pyplot as plt

plt.plot(test_fpr, test_tpr, linestyle = '--', label = 'Patients Last Status')
plt.plot(y_pred_fpr, y_pred_tpr, marker = '.', label = 'Both Sex')
plt.text(0.7, 0.2, "AUC = " + str(y_pred_auc), fontsize = 14)

## Axis lable
plt.xlabel("False Positve Rate")
plt.ylabel("True Positive Rate")

## Show Legend
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


## CONSIDER THE NEURAL NETWORK FOR EACH GENDER SEPARATELY
## Modules required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Code
MBC = (pd.read_excel('MBC.xlsx'))


# In[3]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split
#Import numpy#
import numpy as np


# In[4]:


#### THE MALE DATASET 
my=MBC.PatStatus
mx=MBC.drop(['PatStatus', 'Gender'], axis=1)


# In[5]:


## CONSIDER RBF FITTING FOR THE MALE GENDER


# In[6]:


#Split the Male data into train and test sets #
#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split
#Import numpy#
import numpy as np
mx_train, mx_test, my_train, my_test=train_test_split(mx,my, test_size=0.2, random_state=124)


# In[7]:


mx_train.head()


# In[8]:


mx_test.head()


# In[9]:


mx_train.shape


# In[10]:


mx_test.shape


# In[11]:


## Scaling the male data set
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()
mx_train_minmax = min_max_scaler.fit_transform(mx_train)
mx_test_minmax = min_max_scaler.fit_transform(mx_test)


# In[12]:


mx_train = mx_train_minmax
mx_test = mx_test_minmax


# In[13]:


## FITTING NEURAL NETWORK FOR MALE DATA


# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.model_selection import cross_val_score, cross_validate


# In[15]:


##Fitting the neural network model using training dataset
tns_probs=[0 for _ in range(len(my_test))]


# In[16]:


## Fitting the model
# SVM classifier with Gaussian RBF kernel
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(mx_train,my_train)


# In[17]:


### PREDICTION USING THE TEST DATASET


# In[18]:


### Getting the prediction for the Testing dataset
my_pred = classifier.predict(mx_test)


# In[19]:


## CONFUSION MATRIX FOR MALE DATA
mtest_cm = confusion_matrix(my_test, my_pred)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(mtest_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, mtest_cm[i, j], ha= 'center', va= 'center', color= 'red')
plt.show()


# In[20]:


## Error for the prediction for test dataset outcomes
mtest_error = (mtest_cm[0,1] + mtest_cm[1,0])/np.sum(mtest_cm)
print(mtest_error)


# In[21]:


## Accuracy of prediction
1-mtest_error


# In[22]:


## Sensitivity Analysis 
mtest_sens = mtest_cm[1, 1]/(mtest_cm[1, 1] + mtest_cm[0, 1])
print(mtest_sens)


# In[23]:


## Specificity Analysis
mtest_spec = mtest_cm[0, 0]/(mtest_cm[0, 0]+ mtest_cm[1, 0])
print(mtest_spec)


# In[24]:


## PPV Analysis
mtest_npv = mtest_cm[1, 1]/(mtest_cm[1, 1] + mtest_cm[1, 0])
print(mtest_npv)


# In[25]:


## NPV Analysis
mtest_npv = mtest_cm[0, 0]/(mtest_cm[0, 0] + mtest_cm[0, 1])
print(mtest_npv)


# In[26]:


## The AUC Score
tns_probs=[0 for _ in range(len(my_test))]
mtest_auc = roc_auc_score(my_test, tns_probs)
my_pred_auc = np.round(roc_auc_score(my_test, my_pred), decimals = 2)


# In[27]:


## calculate ROC Curves
mtest_fpr, mtest_tpr, _ = roc_curve(my_test, tns_probs)
my_pred_fpr, my_pred_tpr, _ = roc_curve(my_test, my_pred)


# In[28]:


## Plot Curve for the model
import numpy as np
import matplotlib.pyplot as plt

plt.plot(mtest_fpr, mtest_tpr, linestyle = '--', label = 'Patients Last Status')
plt.plot(my_pred_fpr, my_pred_tpr, marker = '.', label = 'Males')
plt.text(0.7, 0.2, "AUC = " + str(my_pred_auc), fontsize = 14)

## Axis lable
plt.xlabel("False Positve Rate")
plt.ylabel("True Positive Rate")

## Show Legend
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


## CONSIDERING THE FEMALE DATA
## Modules required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
FBC = (pd.read_excel('FBC.xlsx'))

fy=FBC.PatStatus
fx=FBC.drop(['PatStatus','Gender'],axis=1)


# In[2]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

#Import numpy#
import numpy as np
#Split the Male data into train and test sets #
fx_train, fx_test, fy_train, fy_test=train_test_split(fx,fy, test_size=0.2, random_state=125)


# In[3]:


# Scaling the female data
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import numpy as np

min_max_scaler = preprocessing.MinMaxScaler()
fx_train_minmax = min_max_scaler.fit_transform(fx_train)
fx_test_minmax = min_max_scaler.fit_transform(fx_test)


# In[4]:


fx_train = fx_train_minmax
fx_test = fx_test_minmax


# In[5]:


### FITTING THE NEURAL NETWORK USING THE FEMALE TRAINING DATASET 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.model_selection import cross_val_score, cross_validate

tns_probs=[0 for _ in range(len(fy_test))]


# In[6]:


## Fitting the model
# SVM classifier with Gaussian RBF kernel
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(fx_train,fy_train)


# In[7]:


## PREDICTION USING THE TEST DATASET


# In[8]:


### Getting the prediction for the Testing dataset
fy_pred = classifier.predict(fx_test)


# In[9]:


## CONFUSION MATRIX FOR MALE DATA
ftest_cm = confusion_matrix(fy_test, fy_pred)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(ftest_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, ftest_cm[i, j], ha= 'center', va= 'center', color= 'red')
plt.show()


# In[10]:


## Error for the prediction for test dataset outcomes
ftest_error = (ftest_cm[0,1] + ftest_cm[1,0])/np.sum(ftest_cm)
print(ftest_error)


# In[11]:


## Accuracy of prediction
1-ftest_error


# In[12]:


## Sensitivity Analysis 
ftest_sens = ftest_cm[1, 1]/(ftest_cm[1, 1] + ftest_cm[0, 1])
print(ftest_sens)


# In[13]:


## Specificity Analysis
ftest_spec = ftest_cm[0, 0]/(ftest_cm[0, 0]+ ftest_cm[1, 0])
print(ftest_spec)


# In[14]:


## PPV Analysis
ftest_npv = ftest_cm[1, 1]/(ftest_cm[1, 1] + ftest_cm[1, 0])
print(ftest_npv)


# In[15]:


## NPV Analysis
ftest_npv = ftest_cm[0, 0]/(ftest_cm[0, 0] + ftest_cm[0, 1])
print(ftest_npv)


# In[16]:


## The AUC Score
ftest_auc = roc_auc_score(fy_test, tns_probs)
fy_pred_auc = np.round(roc_auc_score(fy_test, fy_pred), decimals = 2)


# In[17]:


## calculate ROC Curves
ftest_fpr, ftest_tpr, _ = roc_curve(fy_test, tns_probs)
fy_pred_fpr, fy_pred_tpr, _ = roc_curve(fy_test, fy_pred)


# In[18]:


## Plot Curve for the model
import numpy as np
import matplotlib.pyplot as plt

plt.plot(ftest_fpr, ftest_tpr, linestyle = '--', label = 'Patients Last Status')
plt.plot(fy_pred_fpr, fy_pred_tpr, marker = '.', label = 'Females')
plt.text(0.7, 0.2, "AUC = " + str(fy_pred_auc), fontsize = 14)

## Axis lable
plt.xlabel("False Positve Rate")
plt.ylabel("True Positive Rate")

## Show Legend
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




