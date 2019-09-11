
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


mybmidata = pd.read_csv(r"C:\Users\Geekquad\Desktop\wac\DataSets-master\DataSets-master\Bmi_male_female.csv")


# In[6]:


mybmidata


# In[8]:


mybmidata.iloc[:,0]


# In[9]:


mybmidata.iloc[:,2:4]


# In[10]:


mybmidata.iloc[2:,2:]


# In[12]:


mybmidata.iloc[::-1,:]


# In[14]:


mybmidata.shape


# # data seperation

# In[20]:


# X_feature=mybmidata.iloc[:,0:3]


# In[19]:


X_feature


# In[36]:


X_feature = mybmidata.iloc[:,:3]
Y_target = mybmidata.iloc[:,3]


# In[38]:


type(X_feature)


# # TRANSFORMATION

# In[63]:


XT_feature= X_feature.values.reshape(500,3)


# In[64]:


XT_feature


# In[65]:


type(XT_feature)


# # similarly

# In[66]:


YT_target = Y_target.values.reshape(500,1)


# In[67]:


X_feature.Gender[X_feature.Gender=="Male"]=1
X_feature.Gender[X_feature.Gender=="Female"]=0


# In[68]:


X_feature


# # data split into train and testing

# In[3]:


X_train = XT_feature[:350]
X_test = XT_feature[350:]
Y_train = YT_target[:350]
Y_test = YT_target[350:]


# In[2]:


X_test.shape()


# In[71]:


index_name= pd.Series(["Extremely Weak","Weak","Normal","Overweight","Obesity","Extremely Obesity"])


# In[72]:


from sklearn.neighbors import KNeighborsClassifier


# In[2]:


teacher = KNeighborsClassifier()


# In[1]:


learner = teacher.fit(X_train,Y_train)


# # testing 

# In[76]:


Ya=Y_test
Yp=learner.predict(X_test)


# In[1]:


from sklearn.metrics import accuracy_score


# In[78]:


acc = accuracy_score(Ya,Yp)*100


# In[79]:


acc

