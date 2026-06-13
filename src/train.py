import pandas
df = pandas.read_csv("data/processed_news.csv")

from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"]).toarray()
y = df["label"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ",accuracy)

# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test, y_pred))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))

import pickle
pickle.dump(model, open("model/fake_news_model.pkl","wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl","wb"))
