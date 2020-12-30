#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
import flask
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# In[42]:


os.chdir(r"E:\Data Science Project\NWS Complaint Management System")


# In[43]:


app = Flask(__name__)

model = pickle.load(open("nlp_model.pkl","rb"))
vector = pickle.load(open("transform.pkl","rb"))


# In[44]:


@app.route('/')

def home():
    return render_template('home.html')


# In[45]:


@app.route('/predict', methods = ["POST"])

def predict():
    
    if request.method == "POST":
        message = request.form['message']
        data = [message]
        data_tfidf = vector.transform(data).toarray()
        my_prediction = model.predict(data_tfidf)
        return render_template('result.html', Prediction = "Category of the complaint is {}".format(my_prediction))
    else:
         print("Please Enter Something")


# In[46]:


if __name__ == "__main__":
    app.debug = True
    app.run()


# In[ ]:




