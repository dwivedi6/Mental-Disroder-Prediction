### Name- Sushil Kumar Dwivedi ####
### Roll No. - MT19AIE323 #####
## The program is for  predicting suicidal thought on twitter data


## Import libraries
import re
import preprocess_kgptalkie as ps
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import tweepy as tw
import pickle


## Data cleaning function
def fn_get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

## Read the Data file
df = pd.read_csv('D:/MTech/Semester VI/MTP/dataset/twitter-suicidal_data.csv')

print("Tweets count against intension")
print(df["intention"].value_counts())

## Clean the tweets
df['tweet']=df['tweet'].apply(lambda x:fn_get_clean(x))

"""
TFIDF score tells the importance of a given word in a given document (when a lot of documents are present). In other words, for a given word query it can actually rank 
the documents wrt. to the relevance with tf-idf score.TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information
retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations (words, phrases, lemmas, etc) in a document amongst a 
collection of documents
"""

tfidf=TfidfVectorizer(max_features=1000,ngram_range=(1,3),analyzer='char')
X=tfidf.fit_transform(df['tweet'])
y=df['intention']


## Split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
## Build the model
clf=LinearSVC()
clf.fit(X_train,y_train)
## Prediction on test data
y_pred=clf.predict(X_test)
print("The classification report")
print(classification_report(y_test,y_pred))

## Save the model ##


with open('D:/MTech/Semester VI/MTP/saved models/mt19aie323-twitter-suicidal-model.pkl', 'wb') as file:
    pickle.dump(clf, file)

### Validation on Twitter Data ####

## Twitter account details
consumer_key= '68IfqPVBvbafzQRaeQJVQmCZB'
consumer_secret= 'hr12LkFFMOwj0z7cmfhMjyXjz7sAkvjmxTmBbXdRp4J5ZhsfqA'
access_token= '706341895551619072-LypLCNjFj8dIeLH7su74fwCZJCSay4G'
access_token_secret= 'nSw2kfNPyRval6lZaaf7kwXGTr939479BMxtT73AsEJbe'

# Authenticate to Twitter
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tw.API(auth)
# test authentication
try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")
    
   
# Get the User object for twitter...
user = api.get_user(screen_name='SushilDwid')
print(user.screen_name)
posts = api.user_timeline(screen_name="SushilDwid",count=100, tweet_mode="extended")

df1=pd.DataFrame([tweet.full_text for tweet in posts] , columns=['Tweets'])

## Clean the tweets
df1['Tweets']=df1['Tweets'].apply(lambda x:fn_get_clean(x))
X1=tfidf.fit_transform(df1['Tweets'])
# Load ML model
model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323-twitter-suicidal-model.pkl', 'rb')) 
#Prediction on tweets
y_pred1=model.predict(X1)
print(y_pred1)
num_correct = 0
for i in range(len(y_pred1)):
    if y_pred1[i] == 1:
        num_correct += 1
print(num_correct)
accuracy=(num_correct/len(y_pred1)*100)
print(accuracy)
if accuracy>8:
  print("Persion is having thinking for suicide attempt")