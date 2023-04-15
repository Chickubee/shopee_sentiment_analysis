# Import library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick
from streamlit_yellowbrick import st_yellowbrick
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from PIL import Image

################
# Import Pakage
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string

def read_sentiment_analysis_model():
    pkl_filename = 'model/lr_model.pkl'
    with open(pkl_filename, 'rb') as file:  
        sentiment_analysis_model = pickle.load(file)
    return sentiment_analysis_model

def read_vectorizer_model():
    pkl_filename = 'model/tfidf_model.pkl'
    with open(pkl_filename, 'rb') as file:  
        vectorizer = pickle.load(file)
    return vectorizer

def predict_sentiment_label(text, vectorizer,sentiment_analysis_model):
    text_features = vectorizer.transform([text])
    predicted_class = sentiment_analysis_model.predict(text_features)[0]   
    if predicted_class == 0:
        negative = Image.open("image/negative.PNG")
        negative = negative.resize((600,600))
        st.image(negative, width = 200)
    elif predicted_class == 1:
        neutral = Image.open("image/neutral.PNG")
        neutral = neutral.resize((600,600))
        st.image(neutral, width = 200)
    elif predicted_class == 2:
        positive = Image.open("image/positive.PNG")
        positive = positive.resize((600,600))
        st.image(positive, width = 200)



# Load model
vectorizer = read_vectorizer_model()
sentiment_analysis_model = read_sentiment_analysis_model()

# GUI
st.markdown("<h1 style='text-align: center; color: grey;'>Sentiment Analyzer Based On Text Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Products Shopee comment </h2>", unsafe_allow_html=True)
st.write('\n\n')

menu = ["1.Business Objective", "2.Build Model", "3.New Prediction"]
choice = option_menu(
    menu_title = None,
    options= menu,
    menu_icon= 'menu-up',
    orientation= 'vertical'
)

if choice == "1.Business Objective":
    # Sentiment Analysis
    st.write("### 1. Sentiment Analysis")
    sentiment_img = Image.open("image/sentiment_analysis.PNG")
    st.image(sentiment_img, width = 700)
    st.markdown("_Sentiment analysis (or opinion mining) is a natural language processing (NLP) technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs._")
 
    # Shopee User Comments Analysis
    st.write("### 2. Shopee User's Comments Analysis")
    st.markdown("_Shopee is an ecosystem 'all-in-one' commercial including shopee.vn. This is a commercial website electronics ranked 1st in Vietnam South and East region South Asia._")
    st.markdown("_From customer's reviews, the problem given is how to the above booth Shopee.vn understands customers better, knows how do they rate themselves to improve products/services quality!_")
    
    
    # Shopee user's comments example
    st.write('\n\n\n')
    st.markdown("**Some user's comments example ...**")
    shopee_comment1 = Image.open("image/shopee_comment1.jpg")
    st.image(shopee_comment1, width = 400)

    shopee_comment2 = Image.open("image/shopee_comment2.jpg")
    st.image(shopee_comment2, width = 400)

    
    # Job 
    st.write("### 3. Some works with this project")
    st.markdown("#### _**3.1. Preprocessing data:**_")
    st.write("     - Convert text to lowercase")
    st.write("     - Remove special characters")
    st.write("     - Replace emojicon/ teencode with corresponding text")
    st.write("     - Replace some punctuation and numbers with spaces")
    st.write("     - Replace misspelled words with spaces")
    st.write("     - Replace series of spaces with a single space")
                
    st.markdown("#### _** 3.2. Standardize Vietnamese Unicode**_")
    st.markdown("#### _** 3.3. Tokenizer Vietnamese text using underthesea library**_")
    st.markdown("#### _** 3.4. Remove Vietnamese stopwords**_")
    st.markdown("#### _** 3.5. Modeling & Evaluation with: Naïve Bayes, Logistic Regression, Random Forest.**_")
    st.markdown("#### _** 3.6. Analyze & Report**_")

    st.write("### 4. Information about author")
    st.write("**Instructors: Ms.Khuat Thuy Phuong**")
    st.write("**Student: Ms.Nguyen Anh Thai**")


elif choice == '2.Build Model':  
    st.write("**Using LOGISTIC REGRESSION model with highest accuracy and short running time to predict sentiment of Shopee's comment. This is the result:**")

    #View some data
    # Information of data
    st.write("### 1. Clean data. View some data after cleaning:")
    data_clean = Image.open("image/data_clean.PNG")
    st.image(data_clean, width = 700)


    st.write("### 2.Confusion Matrix")
    confusion_matrix = Image.open("image/confusion_matrix.PNG")
    st.image(confusion_matrix, width = 700)


    #wordcloud
    st.write("### 3.WordCloud")
    st.write("#### WordCloud plot for label Positive")
    positive_wc = Image.open("image/positive_wc.PNG")
    st.image(positive_wc, width = 700)

    st.write("#### WordCloud plot for label Negative")
    negative_wc = Image.open("image/negative_wc.PNG")
    st.image(negative_wc, width = 700)


    st.write("#### WordCloud plot for label Neutral")
    neutral_wc = Image.open("image/neutral_wc.PNG")
    st.image(neutral_wc, width = 700)
    

elif choice == "3.New Prediction":
    st.subheader("Predict comments's sentiment")
    lines = None
    option = st.radio("",options = ("Input your comment","Upload a comment file"))
    if option == "Input your comment":
        comment = st.text_area("Type your comment here and press ctrl + enter to view result...: ")
        if comment != "":
            comment_cleaned = predict_sentiment_label(comment,vectorizer, sentiment_analysis_model)
    if option == "Upload a comment file":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            df = pd.read_csv(uploaded_file_1, header = None)
            df = df.iloc[1:,:]
            st.write("Your DataFrame:")
            st.dataframe(df[0])
            df['prediction'] = df[0].apply(lambda x: predict_sentiment_label(x,vectorizer, sentiment_analysis_model))
        
           
            


                
