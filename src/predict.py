import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


import pickle
model = pickle.load(open("model/fake_news_model.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))
news = input("Enter news article: ")
clean_news = preprocess(news)
news_vector = vectorizer.transform([clean_news])
prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")