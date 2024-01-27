#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from sklearn.datasets import load_digits


# In[8]:


df = load_digits()


# In[9]:


_, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (10, 3))


# In[10]:


for ax, image, label in zip(axes, df.images, df.target):
    ax.set_axis_off()
    ax.imshow(image, cmap = plt.cm.gray_r, interpolation = "nearest")
    ax.set_title("Training: %i" % label)


# In[11]:


df.images.shape


# In[12]:


df.images[0]


# In[13]:


df.images[0].shape


# In[14]:


len(df.images)


# In[15]:


n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))


# In[16]:


data[0]


# In[17]:


data[0].shape


# In[18]:


data.shape


# In[19]:


data.min()


# In[20]:


data.max()


# In[21]:


data = data/16


# In[22]:


data.min()


# In[23]:


data.max()


# In[24]:


data[0]


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(data, df.target, test_size = 0.3)


# In[27]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


rf = RandomForestClassifier()


# In[30]:


rf.fit(X_train, y_train)


# In[31]:


y_pred = rf.predict(X_test)


# In[32]:


y_pred


# In[33]:


from sklearn.metrics import confusion_matrix, classification_report


# In[34]:


confusion_matrix(y_test, y_pred)


# In[35]:


print(classification_report(y_test, y_pred))


# In[ ]:




