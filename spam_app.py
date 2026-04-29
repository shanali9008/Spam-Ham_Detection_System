import streamlit as st
import nltk
import string 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn as sks
import pickle

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('trained_model.pkr','rb'))


ps = PorterStemmer()  ## we have used this object in function so we have to declair it 

st.title("Spam or Ham Message Detector")

sms = st.text_input("Enter or Paste Message Here")
           
# This is the function for text preprocessings, It will do all preprocessings on the data (Message) we paste in app.
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []    ## empty list to append the text, bcz this func is returning the list
    for i in text:
        if i.isalnum():    ## if there is any special character it will remove it , it will only take alpha numeric values
            y.append(i)

    text = y[:]    # List is mutable datatype , it can not be directry copy, we can just make a clone of it like y[:], [:] says take all the data 
    y.clear()     # clear y list for further storing data

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: 
            y.append(i)

    text= y[:]
    y.clear()

    for i in text: 
        y.append(ps.stem(i))

    return" ".join(y)   # returning the string.. acutally joining whatever inside 'y' to the returning string


transformed_sms=transform_text(sms)

if st.button("Detect"):
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)
    if result == 1:
        st.warning("This is Spam! Report This Sender :warning:")
    else:
        st.success("This not Spam. you are good to Go ", icon=":material/thumb_up:")    

