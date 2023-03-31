import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re


app = Flask(__name__)

# data = pd.read_csv('phishing.csv')
# X = data[['URL']].copy()
# y = data.Label.copy()
# """initialize the label encoder"""
# le = LabelEncoder()
# le.fit(y)

"""Load the saved model & vectorizer"""
le = pickle.load(open('labelencoder.pkl', 'rb'))
rfc = pickle.load(open('rfc_model.pkl', 'rb'))
cv = pickle.load(open("vectorizer.pkl", 'rb')) 

""" Initialize Tokenizer Stemmer and Vectorizer """
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """READ url from the website form and predict"""
    if request.method == 'POST':
        to_predict_list = request.form['review']
        
        review_url=[to_predict_list] 

        url = cv.transform(review_url).toarray() # convert text to bag of words model (Vector)
        phish = rfc.predict(url) # predict Whether the url is good or bad
        phish = le.inverse_transform(phish) # find the url corresponding with the predicted value
        val = phish[0]
        
        return render_template('index.html', prediction='The review is {}'.format(val))


if __name__ == "__main__":
    app.run(debug=True)