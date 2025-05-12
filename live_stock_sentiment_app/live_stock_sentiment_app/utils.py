import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

def get_live_news_sentiment(df, ticker):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []

    for date in df["Date"].dt.date:
        headlines = fetch_news_headlines(ticker, date)
        if headlines:
            score = sum(analyzer.polarity_scores(h)["compound"] for h in headlines) / len(headlines)
        else:
            score = 0
        sentiments.append(score)

    df["Sentiment"] = sentiments
    return df

def fetch_news_headlines(query, date):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&from={date}&to={date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
    )
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        headlines = [article["title"] for article in articles if "title" in article]
        return headlines[:5]  # Limit to top 5
    except Exception as e:
        return []

def train_model(df):
    df = df.dropna()
    X = df[["Sentiment"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    score = r2_score(y, predictions)
    return model, score

def predict_price(model, df):
    latest_sentiment = df["Sentiment"].iloc[-1]
    return model.predict([[latest_sentiment]])[0]