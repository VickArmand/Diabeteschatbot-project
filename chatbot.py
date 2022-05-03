import random,json,pickle,numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lematizer=WordNetLemmatizer()
intents= json.loads(open("intents.json").read())
words= pickle.load(open('words.pkl','rb'))
classes= pickle.load(open('classes.pkl','rb'))
model=load_model('chatbotmodel.model')

def clean_up_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lematizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words=clean_up_sentence(sentence)
    bag=[0]*len(words)