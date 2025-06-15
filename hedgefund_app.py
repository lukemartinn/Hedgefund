# hedgefund_app.py
import os
import yfinance as yf
import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import streamlit as st
from openai.error import RateLimitError

# Page configuration
st.set_page_config(page_title="AI Hedge Fund Analyst", layout="wide")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")
openai_key_env = os.getenv("OPENAI_API_KEY", "")
openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
openai_key = openai_key_input or openai_key_env
if not openai_key:
    st.sidebar.error("Please enter your OpenAI API key above.")
    st.stop()

# Initialize LLM
llm = OpenAI(temperature=0.2, openai_api_key=openai_key)

# Sidebar controls
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper().strip()
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
sma_short = st.sidebar.number_input("Short SMA window", value=20, min_value=1, max_value=200)
sma_long = st.sidebar.number_input("Long SMA window", value=50, min_value=1, max_value=500)
analyze_btn = st.sidebar.button("📊 Run Analysis")

# Functions with caching
@st.cache_data(show_spinner=False)
def get_stock_data(tkr, period, interval, sma1, sma2):
    # Use yfinance history to get a flat DataFrame
    ticker_obj = yf.Ticker(tkr)
    df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
    df[f"SMA_{sma1}"] = df["Close"].rolling(window=sma1).mean()
    df[f"SMA_{sma2}"] = df["Close"].rolling(window=sma2).mean()
    return df

@st.cache_data(show_spinner=False)
def fetch_news(tkr):
    ticker_obj = yf.Ticker(tkr)
    return ticker_obj.news

@st.cache_data(show_spinner=False)
def analyze_with_ai(summary):
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""You are an AI financial analyst. Summarize and provide investment insights based on this data:
{summary}

Also comment on any recent news headlines provided."""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(summary=summary)

# Main layout
st.title("📈 AI Hedge Fund Stock Analysis")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader(f"{ticker} Price Data")
    if analyze_btn:
        try:
            data = get_stock_data(ticker, period, interval, sma_short, sma_long)
            st.line_chart(data[["Close", f"SMA_{sma_short}", f"SMA_{sma_long}"]])
            st.markdown("**Key Statistics:**")
            st.dataframe(data.describe(), hide_index=True)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
    else:
        st.info("Configure parameters in the sidebar and click 'Run Analysis'.")

with col2:
    st.subheader("AI Insights")
    if analyze_btn:
        try:
            summary = data.describe().to_string()
            with st.spinner("Analyzing data with AI..."):
                analysis = analyze_with_ai(summary)
            st.write(analysis)
        except RateLimitError:
            st.error("OpenAI rate limit reached. Please try again later.")
        except Exception as e:
            st.error(f"AI Analysis error: {e}")
    else:
        st.info("AI analysis will appear here.")

st.subheader("📰 Recent News")
if analyze_btn:
    try:
        news = fetch_news(ticker)
        if news:
            for article in news[:5]:
                title = article.get("title", "No Title")
                link = article.get("link", "")
                provider = article.get("publisher", {}).get("name", "")
                st.markdown(f"- [{title}]({link}) — {provider}")
        else:
            st.write("No recent news found.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
