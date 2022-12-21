################### MT19AIE323 #################

## The program is for showing different option for entering user's details for predicting whether the person
# is having desease or not

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import re
import preprocess_kgptalkie as ps
from sklearn.feature_extraction.text import TfidfVectorizer
import tweepy as tw
import pandas as pd
from gensim.models import KeyedVectors
import ftfy
from nltk.corpus import stopwords
from nltk import PorterStemmer
import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np


backgroundColor = '#273346'

# loading the saved models

# Load ML model
survey_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323_mentaldisorderprediction_survey_model.pkl', 'rb')) 
diabetes_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323_diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323_heart_disease_model.sav','rb'))
suicidal_prediction_model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323-twitter-suicidal-model.pkl', 'rb')) 


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Mental Disorder, Suicidal,Diabetes and Heart Desease Prediction System',
                          ['Mental Disorder Prediction on Survey Data',
                           'Suicidal Thought on Twitter Profile',
                           'Depression on Twitter Profile',
                           'Diabetes Prediction',
                           'Heart Disease Prediction'],
                          icons=['person','twitter','twitter','activity','heart'],
                          default_index=0)


# Mental Disorder Prediction on Survey Data
if (selected == 'Mental Disorder Prediction on Survey Data'):
    # page title
    st.title('Mental Disorder Prediction on Survey Data')
    # getting the input data from the user
    col1, col2, col3,col4 = st.columns(4,gap="large")
    
    with col1:

         FHMDoption=('Yes', 'No', 'Do not know')
         FHMD = st.selectbox("Do you have Mental Illness family History?", range(len(FHMDoption)), format_func=lambda x: FHMDoption[x])
        
    with col2:

         CZoption=('1-5', '6-25', '26-100','101-500','501-1000','More than 1000','Do not know')
         CZ = st.selectbox("What is your company size?", range(len(CZoption)), format_func=lambda x: CZoption[x])
    with col3:
         SurveyYoption=('2014', '2016', '2017','2018','2019')
         SurveyY = st.selectbox("What is year of the Survey?", range(len(SurveyYoption)), format_func=lambda x: SurveyYoption[x])
    
    with col4:
        Age = st.text_input('What is your Age?')
    
    with col1:
        if Age<'20':
            AgeG = st.text('0-20')
            AgeG1=1
        elif Age>'20' and Age<='30':
            AgeG = st.text('21-30')
            AgeG1=2
        elif Age>'30' and Age<='40':
            AgeG = st.text('31-40')
            AgeG1=3
        elif Age>'40' and Age<='65':
            AgeG = st.text('41-65')
            AgeG1=4
        elif Age>'65' and Age<='100':
            AgeG = st.text('66-100')
            AgeG1=5
        else:
            AgeG = st.text('66-100')
            AgeG1=6
        
    with col2:
         Genoption=('Male', 'Female', 'Do not disclosed')
         Gen = st.selectbox("What is your gender?", range(len(Genoption)), format_func=lambda x: Genoption[x])
    with col3:

         PreferAnonymityoption=('Yes', 'No','Do not know')
         PreferAnonymity = st.selectbox("Do you Prefer Anonymity?", range(len(PreferAnonymityoption)), format_func=lambda x: PreferAnonymityoption[x]) 
         
    with col4:

         RtPoption=('Above Average', 'Below Average', 'Do not know')
         RtP = st.selectbox("How do you rate Reaction to Problems?", range(len(RtPoption)), format_func=lambda x: RtPoption[x]) 
         
    with col1: 
      
         NegConseqoption=('Yes', 'No', 'May be','Do not know')
         NegConseq = st.selectbox("Do you feel  Negative Consequences?", range(len(NegConseqoption)), format_func=lambda x: NegConseqoption[x]) 
         
    with col2:
        
         AcessToInfooption=('Yes', 'No')
         AcessToInfo = st.selectbox("Do you feel problem in accessing Information?", range(len(AcessToInfooption)), format_func=lambda x: AcessToInfooption[x]) 
         
    with col3: 
         Insuranceoption=('Yes', 'No','Do not know')
         Insurance = st.selectbox("Do you have Insurance?", range(len(Insuranceoption)), format_func=lambda x: Insuranceoption[x]) 
         
    with col4: 
         Diagnosisoption=('Yes', 'No','Sometime','Do not know')
         Diagnosis = st.selectbox("Did you diagnosis Mental Disorder?", range(len(Diagnosisoption)), format_func=lambda x: Diagnosisoption[x]) 
         
    with col1:
         DMHPoption=('Yes', 'No','Maybe','Do not know')
         DMHP = st.selectbox("Is your company is responsible employer?", range(len(DMHPoption)), format_func=lambda x: DMHPoption[x]) 
         
    with col2: 
        ResEmpoption=('Yes', 'No','Maybe','Self-Employed','Do not know')
        ResEmp = st.selectbox("Is your company is responsible employer?", range(len(ResEmpoption)), format_func=lambda x: ResEmpoption[x]) 
        
    with col3:
        Disorderoption=('Yes', 'No')
        Disorder = st.selectbox("Do you have Disorder?", range(len(Disorderoption)), format_func=lambda x: Disorderoption[x]) 
    with col4: 
        PriTechCompoption=('Yes', 'No','Do not know')
        PriTechComp = st.selectbox("Are you working in Sofware Company?", range(len(PriTechCompoption)), format_func=lambda x: PriTechCompoption[x])                            

    
    print(AgeG1)
    # creating a button for Prediction
    survey_diagnosis = ''
    
    print(CZ)
    print(SurveyY)
    print(Age)
    print(AgeG1)
    print(Gen)
    print(PreferAnonymity)
    print(RtP)
    
    if st.button('Mental Disorder Prediction on SurveyData Test Result'): 
        survey_diagnosis = survey_model.predict([[FHMD, CZ, SurveyY, Age, AgeG1, Gen,PreferAnonymity,RtP,NegConseq,AcessToInfo,Insurance,Diagnosis,DMHP,ResEmp,Disorder,PriTechComp]])
        print("predicted figure")
        print(survey_diagnosis)
        if (survey_diagnosis[0] == 1):
          survey_diagnosis = 'The person is having mental Disorder'
        else:
          survey_diagnosis = 'The person is not having mental Disorder'
        
    st.success(survey_diagnosis)

  
# Suicidal Prediction on Tweets
if (selected == 'Suicidal Thought on Twitter Profile'):
    # page title
    st.title('Suicidal Thought on Twitter Profile')
    # getting the input data from the user
    col1, col2, col3 = st.columns(3,gap="large")
    
    with col1:
        screen_name = st.text_input('Screen name')
    with col2:
        consumer_key = st.text_input('consumer key')
    with col3:
        consumer_secret = st.text_input('consumer secret')
    with col1:
        access_token = st.text_input('access token')
    with col2:
        access_token_secret = st.text_input('access token secret')
    
    # code for Prediction
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
    
    
    tweets_diagnosis = ''
    if st.button('Suicidal Prediction result on tweets'):
        # # Authenticate to Twitter
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        api = tw.API(auth)
        # Get the User object for twitter...
        #user = api.get_user(screen_name='SushilDwid')
        posts = api.user_timeline(screen_name=screen_name,count=100, tweet_mode="extended")
        
        tfidf=TfidfVectorizer(max_features=1000,ngram_range=(1,3),analyzer='char')
        df1=pd.DataFrame([tweet.full_text for tweet in posts] , columns=['Tweets'])
        ## Clean the tweets
        df1['Tweets']=df1['Tweets'].apply(lambda x:fn_get_clean(x))
        X1=tfidf.fit_transform(df1['Tweets'])
        #Prediction on tweets
        y_pred1=suicidal_prediction_model.predict(X1)
        num_correct = 0
        for i in range(len(y_pred1)):
            if y_pred1[i] == 1:
                num_correct += 1
        accuracy=(num_correct/len(y_pred1)*100)
        
        if (accuracy>8):
          tweets_diagnosis = 'Person is having suicidal thought'
        else:
          tweets_diagnosis = 'Person is not having suicidal thought'

    st.success(tweets_diagnosis)  


# Depression Prediction on Tweets
if (selected == 'Depression on Twitter Profile'):
    # page title
    st.title('Depression on Twitter Profile')
    # getting the input data from the user
    col1, col2, col3 = st.columns(3,gap="large")
    
    with col1:
        screen_name = st.text_input('Screen name')
    with col2:
        consumer_key = st.text_input('consumer key')
    with col3:
        consumer_secret = st.text_input('consumer secret')
    with col1:
        access_token = st.text_input('access token')
    with col2:
        access_token_secret = st.text_input('access token secret')

    # code for Prediction
    MAX_SEQUENCE_LENGTH = 140 # Max tweet size
    MAX_NB_WORDS = 2000
    EMBEDDING_FILE = 'D:/MTech/Semester VI/MTP/GoogleNews-vectors-negative300.bin.gz'
    #user = api.get_user(screen_name='SushilDwid')
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    # Expand Contraction
    cList = {
      "ain't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "can't've": "cannot have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "I'd": "I would",
      "I'd've": "I would have",
      "I'll": "I will",
      "I'll've": "I will have",
      "I'm": "I am",
      "I've": "I have",
      "isn't": "is not",
      "it'd": "it had",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there had",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'alls": "you alls",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had",
      "you'd've": "you would have",
      "you'll": "you you will",
      "you'll've": "you you will have",
      "you're": "you are",
      "you've": "you have"
    }
    
    c_re = re.compile('(%s)' % '|'.join(cList.keys()))

    def expandContractions(text, c_re=c_re):
        def replace(match):
            return cList[match.group(0)]
        return c_re.sub(replace, text)
    
    def clean_tweets(tweets):
        cleaned_tweets = []
        for tweet in tweets:
            tweet = str(tweet)
            # if url links then dont append to avoid news articles
            # also check tweet length, save those > 10 (length of word "depression")
            if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
                #remove hashtag, @mention, emoji and image URLs
                tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<.>)|(pic\.twitter\.com\/.*)", " ", tweet).split())
                
                #fix weirdly encoded texts
                tweet = ftfy.fix_text(tweet)
                
                #expand contraction
                tweet = expandContractions(tweet)

                #remove punctuation
                tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

                #stop words
                stop_words = set(stopwords.words('english'))
                word_tokens = nltk.word_tokenize(tweet) 
                filtered_sentence = [w for w in word_tokens if not w in stop_words]
                tweet = ' '.join(filtered_sentence)

                #stemming words
                tweet = PorterStemmer().stem(tweet)
                
                cleaned_tweets.append(tweet)

        return cleaned_tweets
 
    depression_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Depression on Twitter Profile'):
        print("Depression Test")
        # Authenticate to Twitter
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        api = tw.API(auth)
        posts = api.user_timeline(screen_name=screen_name,count=100, tweet_mode="extended")
        df1=pd.DataFrame([tweet.full_text for tweet in posts] , columns=['Tweets'])
        depressive_tweets_arr = [x for x in df1['Tweets']]
        X_d = clean_tweets(depressive_tweets_arr)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(X_d)
        sequences_d = tokenizer.texts_to_sequences(X_d)
        data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
        # Load ML model
        model = pickle.load(open('D:/MTech/Semester VI/MTP/saved models/mt19aie323-twitter-depression-model.pkl', 'rb')) 

        labels_pred1 = model.predict(data_d)
        #labels_pred1 = np.round(labels_pred1.flatten())
        num_correct = 0
        for i in range(len(labels_pred1)):
            if labels_pred1[i] == 1:
                num_correct += 1
        accuracy=(num_correct/len(labels_pred1)*100)
        
        if (accuracy>8):
          depression_diagnosis = 'Person''s'' tweets are showing depression thought'
        else:
          depression_diagnosis = 'Person''s'' tweets are not showing depression thought'
  
    st.success(depression_diagnosis)
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Disease Prediction')

    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        

    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    



