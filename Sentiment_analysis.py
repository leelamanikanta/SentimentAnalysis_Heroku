
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[2]:

positive = pd.read_csv('rt-polarity_pos.txt', sep='delimiter', header=None)
negative  = pd.read_csv('rt-polarity_neg.txt', sep='delimiter', header=None)

data_positive = pd.DataFrame({'text':positive[0],'classifcation':1})
data_negative = pd.DataFrame({'text':negative[0],'classifcation':0})


# In[3]:

complete_data = data_positive.append(data_negative, ignore_index = True) 
complete_data['text']= complete_data['text'].str.replace('[^\w\s]',' ')

X = complete_data['text']
y = complete_data['classifcation']


# In[4]:

train_x, valid_x, train_y, valid_y = train_test_split(X,y ,stratify=y,test_size=0.2)


# In[5]:

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000,stop_words='english',ngram_range=(1,3), min_df=10)
tfidf_vect.fit(train_x)


# In[6]:

pickle.dump(tfidf_vect, open('tfid.pkl','wb'))


# In[7]:

xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)


# In[8]:

model = LogisticRegression()
model.fit(xtrain_tfidf, train_y)
pickle.dump(model, open('model.pkl','wb'))
accuracy = accuracy_score(model.predict(xvalid_tfidf), valid_y)
print ("Accuracy: ", accuracy)


# In[ ]:




# In[2]:

m = pickle.load(open('model.pkl','rb'))
tf = pickle.load(open('tfid.pkl','rb'))


# In[3]:

x= tf.transform(['bad'])
output = m.predict(x)[0]
mapping = dict({1:'positive',0:'negative'})
mapping[output]


# In[ ]:



