import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("fake-news-detection\data\processed_news.csv")

df["content"] = df["content"].str.lower()

df["content"] = df["content"].apply(
    lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
)

stop_words = set(stopwords.words('english'))
df["content"] = df["content"].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"]).toarray()
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)