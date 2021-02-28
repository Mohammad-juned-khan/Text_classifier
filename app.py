# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import requests
import pickle
import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wordnet = WordNetLemmatizer()

app = Flask(__name__)
modell = pickle.load(open('final_model', 'rb'))
tfidf = pickle.load(open('tfidf', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('Index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        text = (request.form['article'])
        def preprocess_text(text):
            text= text.lower()
            text = text.replace('\n', ' ')
            text = text.replace('\r', '')
            text = text.strip()
            text = re.sub(' +', ' ', text)
            text = re.sub(r'[^\w\s]', '', text)

            # removing stop words
            word_tokens = word_tokenize(text)
            filtered_sentence = []
            for word in word_tokens:
                if word not in set(stopwords.words('english')):
                    filtered_sentence.append(word)

            text = " ".join(filtered_sentence)

            return text

        def input_predict(text):
            # preprocess the text
            text = preprocess_text(text)
            # convert text to a list
            yh = [text]
            # transform the input
            inputpredict = tfidf.transform(yh)
            # predict the user input text
            y_predict_userinput = modell.predict(inputpredict)

            return y_predict_userinput
        output = input_predict(text)
        cat = int(output)
        if cat == 0:
            category ="Business"
        elif cat == 1:
            category ="Entertainment"
        elif cat == 2:
            category ="Politics"
        elif cat == 3:
            category ="Sports"
        elif cat == 4:
            category ="Technology"
        else:
            category = "Error"

        return render_template('Result.html', text="Your text is based on {}. ".format(category))

if __name__=="__main__":
    app.run(debug=True)
