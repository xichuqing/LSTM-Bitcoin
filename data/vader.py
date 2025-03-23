from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

#%%
root = Path(__file__).resolve().parent.parent
tweets_path = os.path.join( root,"data","mbsa.csv")
trade_path = os.path.join(root,"data","BTC-USD.csv")

df2 = pd.read_csv(tweets_path)
trade_data = pd.read_csv(trade_path)

start_date = datetime.strptime("2017-06-25", "%Y-%m-%d").date()
end_date = datetime.strptime("2019-06-25", "%Y-%m-%d").date()

df2 = df2[["Date","text"]]
df2["Date"] = pd.to_datetime(df2["Date"], errors='coerce', utc=True)
df2["Date"] = df2["Date"].dt.date
df2 = df2[(df2["Date"] >= start_date) & (df2["Date"] <= end_date)]
df2 = df2.drop_duplicates()
df2["text"]  = df2["text"].astype(str) 
df2['text'] = df2['text'].str.lower()
df2['text'] = df2['text'].str.replace("@[A-Za-z0-9_]+","")
df2['text'] = df2['text'].str.replace("#[A-Za-z0-9_]+","")
df2['text'] = df2['text'].str.replace(r"http\S+", "")
df2['text'] = df2['text'].str.replace(r"www.\S+", "")
df2['text'] = df2['text'].str.replace('[()!?]', ' ')
df2['text'] = df2['text'].str.replace('\[.*?\]',' ')
df2['text'] = df2['text'].str.replace("[^a-z0-9]"," ")

trade_data["Date"] = pd.to_datetime(trade_data["Date"])
trade_data.set_index("Date", inplace=True)
trade_data = trade_data.truncate(before=start_date, after=end_date)  

#%%
analyzer = SentimentIntensityAnalyzer()

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df2["polarity"] = df2['text'].apply(getPolarity)
df2["subjectivity"] = df2['text'].apply(getSubjectivity)
df2 = df2[df2['polarity'] != 0]
#%%
polarity_daily_avg = df2.groupby('Date')['polarity'].mean().rename("polarity").reset_index()
subjectivity = df2.groupby('Date')['subjectivity'].mean().rename("subjectivity").reset_index()
polarity_daily_avg['Date'] = pd.to_datetime(polarity_daily_avg['Date'])
subjectivity['Date'] = pd.to_datetime(subjectivity['Date'])
#%%
df = trade_data.merge(polarity_daily_avg, left_on='Date', right_on = 'Date')
df = df.merge(subjectivity, left_on='Date',right_on = 'Date')
df.to_csv("df.csv",index = False)
# %%
