import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


message_data = pd.read_csv("spam.csv",encoding = "latin")

message_data.head()

message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})

message_data.insert(0, 'New_ID', range(0, 0 + len(message_data)))

message_data.groupby('Spam/Not_Spam').describe()

print("message_data")
print(message_data)
print("-------------------------------------------------------------------------------------")

message_data_copy = message_data['message'].copy()

def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

message_data_copy = message_data_copy.apply(text_preprocess)


print("Message data copy after textprocessing")
print(message_data_copy)
print("-------------------------------------------------------------------------------------")

vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)
message_mat

message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)

print("accuracyscore after textprocessing")
print(accuracy_score(spam_nospam_test,pred))
print("------------------------------------------------------------------------------------")

def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

message_data_copy = message_data_copy.apply(stemmer)

print("messagedatacopy after stemming")
print(message_data_copy)
print("-------------------------------------------------------------------------------------")

vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)

message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat,
                                                        message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)

print("Accuracy after stemming")
print(accuracy_score(spam_nospam_test, pred))
print("-------------------------------------------------------------------------------------")

message_data['length'] = message_data['message'].apply(len)
length = message_data['length'].to_numpy()
new_mat = np.hstack((message_mat.todense(),length[:, None]))
message_train, message_test, spam_nospam_train, spam_nospam_test, id_train, id_test = train_test_split(new_mat,
                                                        message_data['Spam/Not_Spam'], message_data['New_ID'], test_size=0.3, random_state=20)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)
pred = Spam_model.predict(message_test)
accuracy_score(spam_nospam_test,pred)

print("Accuracy score after normalising length")
print(accuracy_score(spam_nospam_test, pred))
print("--------------------------------------------------------------------------------------")

# Second level of classification - Doing sentiment analysis on spam and ham messages

from textblob import TextBlob

count=0
count1=0
spam_positive=0
spam_negative=0
spam_neutral=0
not_spam_positive=0
not_spam_negative=0
not_spam_neutral=0

spamtext = ''
notspamtext = ''
positivetext = ''
negativetext = ''
neutraltext = ''

spampositive = ''
spamnegative = ''
spamneutral = ''
hampositive = ''
hamnegative = ''
hamneutral = ''


a = 0
n = 0


'''#To check how many messages are predicted correctly
for sentence in message_test:
    pred = Spam_model.predict(sentence)
    if pred != spam_nospam_test.iloc[n]:
        a = a + 1
    n = n + 1

print("prediction failed for  " + str(a) + " test messages")
print("total test messages  " + str(n))'''

a = 0
n = 0
for sentence in message_test:
    pred = Spam_model.predict(sentence)
    if pred == 'spam':
        #print(message_data.iloc[id_test.iloc[n]]['message'])
        #print(spam_nospam_test.iloc[n])
        spamtext = spamtext + message_data.iloc[id_test.iloc[n]]['message']
        blob = TextBlob(message_data.iloc[id_test.iloc[n]]['message'])
       #print(blob.sentiment.polarity)
        if blob.sentiment.polarity > 0.0:
            spampositive = spampositive + message_data.iloc[id_test.iloc[n]]['message']
            positivetext = positivetext + message_data.iloc[id_test.iloc[n]]['message']
            spam_positive = spam_positive + 1
        elif blob.sentiment.polarity < 0.0:
            spamnegative = spamnegative + message_data.iloc[id_test.iloc[n]]['message']
            negativetext = negativetext + message_data.iloc[id_test.iloc[n]]['message']
            spam_negative = spam_negative + 1
        else:
            spamneutral = spamneutral + message_data.iloc[id_test.iloc[n]]['message']
            neutraltext = neutraltext + message_data.iloc[id_test.iloc[n]]['message']
            spam_neutral = spam_neutral + 1
    elif pred == 'ham':
        #print(message_data.iloc[id_test.iloc[n]]['message'])
        #print(spam_nospam_test.iloc[n])
        notspamtext = notspamtext + message_data.iloc[id_test.iloc[n]]['message']
        blob = TextBlob(message_data.iloc[id_test.iloc[n]]['message'])
        #print(blob.sentiment.polarity)
        if blob.sentiment.polarity > 0.0:
            hampositive = hampositive + message_data.iloc[id_test.iloc[n]]['message']
            positivetext = positivetext + message_data.iloc[id_test.iloc[n]]['message']
            not_spam_positive = not_spam_positive + 1
        elif blob.sentiment.polarity < 0.0:
            hamnegative = hamnegative + message_data.iloc[id_test.iloc[n]]['message']
            negativetext = negativetext + message_data.iloc[id_test.iloc[n]]['message']
            not_spam_negative = not_spam_negative + 1
        else:
            hamneutral = hamneutral + message_data.iloc[id_test.iloc[n]]['message']
            neutraltext = neutraltext + message_data.iloc[id_test.iloc[n]]['message']
            not_spam_neutral = not_spam_neutral + 1
    n = n + 1


print('Spam Positive ' + str(spam_positive))
print('Spam Negative ' + str(spam_negative))
print('Spam Neutral ' + str(spam_neutral))
print("Percentage of spam positives among spam " + str((spam_positive * 100)/(spam_positive + spam_negative + spam_neutral)))
print("Percentage of spam negatives among spam " + str((spam_negative * 100)/(spam_positive + spam_negative + spam_neutral)))
print("Percentage of spam neutrals among spam " + str((spam_neutral * 100)/(spam_positive + spam_negative + spam_neutral)))

print("----------------------------------------------------------------------------------------------------------------------")

print('Not Spam Positive ' + str(not_spam_positive))
print('Not Spam Negative ' + str(not_spam_negative))
print('Not Spam Neutral ' + str(not_spam_neutral))
print("Percentage of not spam positives among spam " + str((not_spam_positive * 100)/(not_spam_positive + not_spam_negative + not_spam_neutral)))
print("Percentage of not spam negatives among spam " + str((not_spam_negative * 100)/(not_spam_positive + not_spam_negative + not_spam_neutral)))
print("Percentage of not spam neutrals among spam " + str((not_spam_neutral * 100)/(not_spam_positive + not_spam_negative + not_spam_neutral)))


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

wordcloud = WordCloud().generate(spampositive)
wordcloud.to_file("img/spampositive.png")

wordcloud = WordCloud().generate(spamnegative)
wordcloud.to_file("img/spamnegative.png")

wordcloud = WordCloud().generate(spamneutral)
wordcloud.to_file("img/spamneutral.png")

wordcloud = WordCloud().generate(hampositive)
wordcloud.to_file("img/hampositive.png")

wordcloud = WordCloud().generate(hamnegative)
wordcloud.to_file("img/hamnegative.png")

wordcloud = WordCloud().generate(hamneutral)
wordcloud.to_file("img/hamneutral.png")
