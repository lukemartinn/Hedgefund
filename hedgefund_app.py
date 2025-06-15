# hedgefund_app.py
import os
import yfinance as yf
import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import streamlit as st

# Function to fetch stock data
def get_stock_data(ticker: str, period: str='1y', interval: str='1d') -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

# Initialize LLM
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
llm = OpenAI(temperature=0, openai_api_key=openai_key)

def analyze_with_llm(summary: str) -> str:
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
You are an AI financial analyst. Here's the summary of stock data:
{summary}

Provide a brief analysis and recommendation.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(summary=summary)

def fetch_news_via_yf(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.news

# Streamlit App
st.title("AI Hedge Fund Stock Analysis")
ticker = st.text_input("Enter Stock Ticker:", value="AAPL")

if st.button("Analyze"):
    data = get_stock_data(ticker)
    st.subheader("Stock Data")
    st.line_chart(data[['Close', 'SMA_20', 'SMA_50']])
    st.subheader("Data Summary")
    st.write(data.describe())
    with st.spinner("Analyzing with AI..."):
        analysis = analyze_with_llm(data.describe().to_string())
        st.subheader("AI Analysis")
        st.write(analysis)
    st.subheader("Recent News")
    news = fetch_news_via_yf(ticker)
    st.json(news)
