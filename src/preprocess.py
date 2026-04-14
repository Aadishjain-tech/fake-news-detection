import pandas as pd

fake = pd.read_csv("fake-news-detection\data\Fake.csv")
true = pd.read_csv("fake-news-detection\data\True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1)
df = df.reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]

df.to_csv("fake-news-detection\data\processed_news.csv", index=False)