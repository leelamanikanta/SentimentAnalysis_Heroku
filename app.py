
# coding: utf-8

# In[3]:

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import random
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfid.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    output = ""
    features = list(request.form.values())
    x= tfidf.transform(features)
    output = model.predict(x)[0]
    mapping = dict({1:'positive',0:'negative'})
    output = mapping[output]
    return render_template('index.html', prediction_text='Sentiment: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:



