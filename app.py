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


@app.route('/', methods=['GET'])
def home():
    return render_template('Index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        input_text = (request.form['article'])

        def preprocess_text(input_text):
            text_lower = input_text.lower()
            text_replaced = text_lower.replace('\n', ' ')
            text1 = text_replaced.replace('\r', '')
            text_strip = text1.strip()
            text2 = re.sub(' +', ' ', text_strip)
            text3 = re.sub(r'[^\w\s]', '', text2)

            # removing stop words
            word_tokens = word_tokenize(text3)
            filtered_sentence = []
            for word in word_tokens:
                if word not in set(stopwords.words('english')):
                    filtered_sentence.append(word)

            text3 = " ".join(filtered_sentence)

            return text3

        def input_predict(input_text):
            # preprocess the text
            processed_text = preprocess_text(input_text)
            # convert processed text to a list
            yh = [processed_text]
            # transform the input text to vectors
            inputpredict = tfidf.transform(yh)
            # predict the user input text
            y_predict_userinput = modell.predict(inputpredict)

            return y_predict_userinput
        
        output = input_predict(input_text)
        cat = int(output)
        if cat == 0:
            category = "Business"
        elif cat == 1:
            category = "Entertainment"
        elif cat == 2:
            category = "Politics"
        elif cat == 3:
            category = "Sports"
        elif cat == 4:
            category = "Technology"
        else:
            category = "Error"

        return render_template('Index.html', text="Your text is based on {}. ".format(category))
    else:
        return render_template('Index.html')


if __name__ == "__main__":
    app.run(debug=True)
