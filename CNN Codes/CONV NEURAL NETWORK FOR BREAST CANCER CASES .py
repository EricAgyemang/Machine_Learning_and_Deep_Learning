#!/usr/bin/env python
# coding: utf-8

# In[1]:


### BREAST CANCER CASES ###
###### CONVOLUTIONAL NEURAL NETWORK CODE IN JUPYTER NOTEBOOK FOR BOTH SEX #####


# In[2]:


## Modules required
import pandas as pd
import numpy as np
from scipy import misc 
from PIL import Image 
import glob 
from matplotlib.pyplot import imshow 
import seaborn as sn
import pickle 
from keras.preprocessing import image 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
from keras.utils import layer_utils, np_utils 
from keras.utils.data_utils import get_file 
from keras.applications.imagenet_utils import preprocess_input 
from keras.utils.vis_utils import model_to_dot 
from keras.utils import plot_model 
from keras.initializers import glorot_uniform 
from keras import losses 
import keras.backend as K 
from keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix, classification_report 
from keras import layers 
from IPython.display import SVG
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Input, Add, ZeroPadding2D, Conv2D, MaxPooling2D 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# In[3]:


# Code
BC = (pd.read_excel('cancer.xlsx'))


# In[4]:


BC.head()


# In[5]:


## Reshaping into array
import random
random.seed(30)
BC.iloc[3,1:].values.reshape(6,4).astype('int8')


# In[6]:


## Preprocessing the data


# In[7]:


## Storing the independent variables array in form length, width, height into df_x
random.seed(31)
df_x = BC.iloc[:,1:].values.reshape(len(BC), 6, 4, 1)

## Storing the dependent variables in y
y = BC.iloc[:,0].values


# In[8]:


# converting y to categorical
df_y = keras.utils.to_categorical(y, num_classes = 2)


# In[9]:


df_x =np.array(df_x)
df_y = np.array(df_y)


# In[10]:


df_y


# In[11]:


df_x.shape


# In[12]:


df_y.shape


# In[13]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

#Import numpy#
import numpy as np
random.seed(32)
#Split the data into train and test sets #
x_train, x_test, y_train, y_test=train_test_split(df_x,df_y, test_size=0.2, random_state=123)


# In[14]:


x_test.shape


# In[15]:


y_test.shape


# In[16]:


### CNN Model
random.seed(33)
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = (6, 4, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.25)) 

model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[17]:


model.summary()


# In[18]:


## fitting the model with 
CNN_MODEL = model.fit(x_train, y_train, batch_size=40, epochs=10, validation_data=(x_test, y_test))


# In[19]:


## MODEL EVALUATION FOR BOTH SEX


# In[20]:


## Prediction loss and accuracy
test_eval = model.evaluate(x_test, y_test, verbose=0)[1]


# In[21]:


print('Test accuracy:', test_eval)


# In[22]:


##plot the accuracy and loss plots between training and validation data to check for over-fitting

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

accuracy = CNN_MODEL.history['accuracy']
val_accuracy = CNN_MODEL.history['val_accuracy']
loss = CNN_MODEL.history['loss']
val_loss = CNN_MODEL.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[23]:


##plot our training accuracy and validation accuracy
plt.plot(CNN_MODEL.history['accuracy'])
plt.plot(CNN_MODEL.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[24]:


## Predicting using CNN
CNN_MODEL_pred = model.predict(x_test, batch_size=32, verbose=1)
CNN_MODEL_predicted = np.argmax(CNN_MODEL_pred, axis=1)


# In[25]:


## Confusion matrix for the CNN
CNN_MODEL_cm = confusion_matrix(np.argmax(y_test, axis=1), CNN_MODEL_predicted)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(CNN_MODEL_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, CNN_MODEL_cm[i, j], ha= 'center', va= 'center', color= 'red')
plt.show()


# In[26]:


test_cm = CNN_MODEL_cm


# In[27]:


## Sensitivity Analysis 
test_sens = test_cm[1, 1]/(test_cm[1, 1] + test_cm[0, 1])
print(test_sens)


# In[28]:


## Specificity Analysis
test_spec = test_cm[0, 0]/(test_cm[0, 0]+test_cm[1, 0])
print(test_spec)


# In[29]:


## PPV Analysis
test_npv = test_cm[1, 1]/(test_cm[1, 1] + test_cm[1, 0])
print(test_npv)


# In[30]:


## NPV Analysis
test_npv = test_cm[0, 0]/(test_cm[0, 0]+test_cm[0, 1])
print(test_npv)


# In[31]:


from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

n_classes = 1

from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 8

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], CNN_MODEL_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), CNN_MODEL_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Both Sex (AUC = {0:0.1f})'
               ''.format(roc_auc["micro"]),marker = '.',
         color='orange', linestyle=':', linewidth=2)

plt.plot([0, 1], [0, 1], 'b--', label = 'Patients Last Status',linewidth=2, lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


## CONSIDER THE CONVOLUTIONAL NEURAL NETWORK FOR EACH GENDER SEPARATELY


# In[2]:


## Modules required
import pandas as pd
import numpy as np
from scipy import misc 
from PIL import Image 
import glob 
from matplotlib.pyplot import imshow 
import seaborn as sn
import pickle 
from keras.preprocessing import image 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
from keras.utils import layer_utils, np_utils 
from keras.utils.data_utils import get_file 
from keras.applications.imagenet_utils import preprocess_input 
from keras.utils.vis_utils import model_to_dot 
from keras.utils import plot_model 
from keras.initializers import glorot_uniform 
from keras import losses 
import keras.backend as K 
from keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix, classification_report 
from keras import layers 
from IPython.display import SVG
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Input, Add, ZeroPadding2D, Conv2D, MaxPooling2D 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# In[3]:


# Code
MBC = (pd.read_excel('MBC.xlsx'))


# In[4]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split
#Import numpy#
import numpy as np


# In[5]:


## CONSIDER FITTING CONVOLUTIONAL NEURAL NETWORK FOR THE MALE GENDER


# In[6]:


MBC.head()


# In[7]:


## Reshaping into array
MBC.iloc[3,1:].values.reshape(6,4).astype('int8')


# In[8]:


## Preprocessing the data


# In[9]:


## Storing the independent variables array in form length, width, height into df_x
df_x = MBC.iloc[:,1:].values.reshape(len(MBC), 6, 4, 1)

## Storing the dependent variables in y
y = MBC.iloc[:,0].values


# In[10]:


# converting y to categorical
df_y = keras.utils.to_categorical(y, num_classes = 2)


# In[11]:


df_x =np.array(df_x)
df_y = np.array(df_y)


# In[12]:


df_y


# In[13]:


df_x.shape


# In[14]:


df_y.shape


# In[15]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

#Import numpy#
import numpy as np
#Split the data into train and test sets #
x_train, x_test, y_train, y_test=train_test_split(df_x,df_y, test_size=0.2, random_state=123)


# In[16]:


x_test.shape


# In[17]:


y_test.shape


# In[18]:


### CNN Model
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = (6, 4, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.25)) 

model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[19]:


model.summary()


# In[20]:


## fitting the model with 
CNN_MODEL = model.fit(x_train, y_train, batch_size=40, epochs=10, validation_data=(x_test, y_test))


# In[21]:


## MODEL EVALUATION


# In[22]:


## Prediction loss and accuracy
test_eval = model.evaluate(x_test, y_test, verbose=0)[1]


# In[23]:


print('Test accuracy:', test_eval)


# In[24]:


##plot the accuracy and loss plots between training and validation data to check for over-fitting

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

accuracy = CNN_MODEL.history['accuracy']
val_accuracy = CNN_MODEL.history['val_accuracy']
loss = CNN_MODEL.history['loss']
val_loss = CNN_MODEL.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[25]:


##plot our training accuracy and validation accuracy
plt.plot(CNN_MODEL.history['accuracy'])
plt.plot(CNN_MODEL.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[26]:


## Predicting using CNN
CNN_MODEL_pred = model.predict(x_test, batch_size=32, verbose=1)
CNN_MODEL_predicted = np.argmax(CNN_MODEL_pred, axis=1)


# In[27]:


## Confusion matrix for the CNN
CNN_MODEL_cm = confusion_matrix(np.argmax(y_test, axis=1), CNN_MODEL_predicted)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(CNN_MODEL_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, CNN_MODEL_cm[i, j], ha= 'center', va= 'center', color= 'k')
plt.show()


# In[28]:


test_cm = CNN_MODEL_cm


# In[29]:


## Sensitivity Analysis 
test_sens = test_cm[1, 1]/(test_cm[1, 1] + test_cm[0, 1])
print(test_sens)


# In[30]:


## Specificity Analysis
test_spec = test_cm[0, 0]/(test_cm[0, 0]+test_cm[1, 0])
print(test_spec)


# In[31]:


## PPV Analysis
test_npv = test_cm[1, 1]/(test_cm[1, 1] + test_cm[1, 0])
print(test_npv)


# In[32]:


## NPV Analysis
test_npv = test_cm[0, 0]/(test_cm[0, 0]+test_cm[0, 1])
print(test_npv)


# In[33]:


from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

n_classes = 1

from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 8

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], CNN_MODEL_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), CNN_MODEL_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Male (AUC = {0:0.2f})'
               ''.format(roc_auc["micro"]),marker = '.',
         color='orange', linestyle=':', linewidth=2)

plt.plot([0, 1], [0, 1], 'b--', label = 'Patients Last Status',linewidth=2, lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


## CONSIDERING THE FEMALE DATA SEPARATELY FOR THE ANALYSIS


# In[2]:


## Modules required
import pandas as pd
import numpy as np
from scipy import misc 
from PIL import Image 
import glob 
from matplotlib.pyplot import imshow 
import seaborn as sn
import pickle 
from keras.preprocessing import image 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
from keras.utils import layer_utils, np_utils 
from keras.utils.data_utils import get_file 
from keras.applications.imagenet_utils import preprocess_input 
from keras.utils.vis_utils import model_to_dot 
from keras.utils import plot_model 
from keras.initializers import glorot_uniform 
from keras import losses 
import keras.backend as K 
from keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix, classification_report 
from keras import layers 
from IPython.display import SVG
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Input, Add, ZeroPadding2D, Conv2D, MaxPooling2D 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


# In[3]:


# Code
FBC = (pd.read_excel('FBC.xlsx'))


# In[4]:


## CONSIDERING THE FEMALE DATA SEPARATELY FOR THE ANALYSIS
FBC.head()


# In[5]:


## The new fitted logistic regression model with selected variables
## Modules required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten, Conv2D, MaxPooling2D 
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

## Reshaping into array
FBC.iloc[3,1:].values.reshape(6,4).astype('int8')


# In[6]:


## Preprocessing the data


# In[7]:


## Storing the independent variables array in form length, width, height into df_x
df_x = FBC.iloc[:,1:].values.reshape(len(FBC), 6, 4, 1)

## Storing the dependent variables in y
y = FBC.iloc[:,0].values


# In[8]:


# converting y to categorical
df_y = keras.utils.to_categorical(y, num_classes = 2)


# In[9]:


df_x =np.array(df_x)
df_y = np.array(df_y)


# In[10]:


df_y


# In[11]:


df_x.shape


# In[12]:


df_y.shape


# In[13]:


#Import 'train_test_split' from 'sklearn.model_selection'
from sklearn.model_selection import train_test_split

#Import numpy#
import numpy as np
#Split the data into train and test sets #
x_train, x_test, y_train, y_test=train_test_split(df_x,df_y, test_size=0.2, random_state=123)


# In[14]:


x_test.shape


# In[15]:


y_test.shape


# In[16]:


### CNN Model
model = Sequential()
model.add(Conv2D(64, (2,3), input_shape = (6, 4, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.25)) 

model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


# In[17]:


model.summary()


# In[18]:


## fitting the model with 
CNN_MODEL = model.fit(x_train, y_train, batch_size=30, epochs=10, validation_data=(x_test, y_test))


# In[19]:


## MODEL EVALUATION


# In[20]:


## Prediction loss and accuracy
test_eval = model.evaluate(x_test, y_test, verbose=0)[1]


# In[21]:


print('Test accuracy:', test_eval)


# In[22]:


##plot the accuracy and loss plots between training and validation data to check for over-fitting

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

accuracy = CNN_MODEL.history['accuracy']
val_accuracy = CNN_MODEL.history['val_accuracy']
loss = CNN_MODEL.history['loss']
val_loss = CNN_MODEL.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[23]:


##plot our training accuracy and validation accuracy
plt.plot(CNN_MODEL.history['accuracy'])
plt.plot(CNN_MODEL.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[24]:


## Predicting using CNN
CNN_MODEL_pred = model.predict(x_test, batch_size=32, verbose=1)
CNN_MODEL_predicted = np.argmax(CNN_MODEL_pred, axis=1)


# In[25]:


## Confusion matrix for the CNN
CNN_MODEL_cm = confusion_matrix(np.argmax(y_test, axis=1), CNN_MODEL_predicted)
fig, ax = plt.subplots(figsize = (8, 8))
ax.imshow(CNN_MODEL_cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1), ticklabels=('Actual 1s', 'Actual 0s'))
ax.yaxis.set(ticks=(0,1), ticklabels=('predicted 1s', 'predicted 0s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, CNN_MODEL_cm[i, j], ha= 'center', va= 'center', color= 'k')
plt.show()


# In[26]:


test_cm = CNN_MODEL_cm


# In[27]:


## Sensitivity Analysis 
test_sens = test_cm[1, 1]/(test_cm[1, 1] + test_cm[0, 1])
print(test_sens)


# In[28]:


## Specificity Analysis
test_spec = test_cm[0, 0]/(test_cm[0, 0]+test_cm[1, 0])
print(test_spec)


# In[29]:


## PPV Analysis
test_npv = test_cm[1, 1]/(test_cm[1, 1] + test_cm[1, 0])
print(test_npv)


# In[30]:


## NPV Analysis
test_npv = test_cm[0, 0]/(test_cm[0, 0]+test_cm[0, 1])
print(test_npv)


# In[32]:


from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

n_classes = 1

from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 8

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], CNN_MODEL_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), CNN_MODEL_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='Female (AUC = {0:0.1f})'
               ''.format(roc_auc["micro"]),marker = '.',
         color='orange', linestyle=':', linewidth=2)

plt.plot([0, 1], [0, 1], 'b--', label = 'Patients Last Status',linewidth=2, lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




