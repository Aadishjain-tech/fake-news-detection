import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([true,fake],axis=0)
df = df.sample(frac=1)
df = df.reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]


df["content"] = df["content"].str.lower()

import re
df["content"] =  df["content"].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))

from nltk.corpus import stopwords
import nltk
stop_words = set(stopwords.words('english'))
df["content"] = df["content"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

df.to_csv("data/processed_news.csv", index=False)
