import streamlit as st
from utils import get_live_news_sentiment, get_stock_data, train_model, predict_price
import datetime

st.set_page_config(page_title="Live Stock Sentiment Prediction", layout="centered")

st.title("📈 Live Stock Price Prediction using Sentiment Analysis")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

if st.button("Predict"):
    with st.spinner("Fetching data and predicting..."):
        df = get_stock_data(ticker, start_date, end_date)
        df = get_live_news_sentiment(df, ticker)
        model, score = train_model(df)
        predicted_price = predict_price(model, df)

        st.subheader("📊 Predicted Next Close Price")
        st.metric(label=f"{ticker} Prediction", value=f"${predicted_price[0]:.2f}")

        st.subheader("🔍 Model Performance")
        st.write(f"R² Score: {score:.2f}")

        st.line_chart(df.set_index("Date")[["Close"]])
