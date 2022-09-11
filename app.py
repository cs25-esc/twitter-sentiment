import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import re
import seaborn as sns

import matplotlib.pyplot as plt

from textblob import TextBlob
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stw = PorterStemmer()

stop_words = set(stopwords.words('english'))


def data_pre(text):
  text = text.lower()
  text = re.sub(r"https\S+","",text,flags=re.MULTILINE)

  text = re.sub(r'\@w+|\#',"",text)

  text = re.sub(r'[^\w\s]' , "", text)

  text_tokens = word_tokenize(text)

  filtered_text = [w for w in text_tokens if not w in stop_words]

  wer = [stw.stem(i) for i in filtered_text]


  return " ".join(wer)

def polarity(text):
    p = TextBlob(text).sentiment.polarity

    if p < 0:
        return 'Negative'
    elif p > 0:
        return 'Positive'
    else:
        return 'Neutral'


# Creating list to append tweet data to

st.title("Twitter Sentiment Analysis")

keyword =    st.text_input('Enter the keyword')

a =  "#" + keyword
b = keyword

m  =  a

n = st.text_input('Enter no of tweets')

if  st.button('Extract tweets & Analyze'):
    tweets_list2 = []

        # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(m).get_items()):

        if i > int(n):
            break
        tweets_list2.append([tweet.date, tweet.content])

        # Creating a dataframe from the tweets list above
    tw = pd.DataFrame(tweets_list2, columns=['Datetime', 'Text'])
        #streamlit run app.py

    tw['Text'] = tw['Text'].apply(data_pre)

    tw['sentiment'] = tw['Text'].apply(polarity)

    selected_cust_id = st.selectbox(
                'select your cust id' ,
                tw['sentiment'].values)


    fig,ax = plt.subplots()

    ax.bar(tw.sentiment.values , tw.index)

    st.pyplot(fig)

    pos = len(tw[tw['sentiment'] == 'Positive']) / len(tw)
    neg = len(tw[tw['sentiment'] == 'Negative']) / len(tw)
    neu = len(tw[tw['sentiment'] == 'Neutral']) / len(tw)

    st.text(pos)
    st.text(neg)
    st.text(neu)