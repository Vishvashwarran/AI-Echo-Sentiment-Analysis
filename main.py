import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from wordcloud import WordCloud
import spacy
import re

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    layout="wide"
)

st.title("🤖 AI Echo: Your Smartest Conversational Partner")
st.write(
    "Sentiment Analysis of ChatGPT User Reviews using NLP, "
    "Machine Learning, and Deep Learning"
)

# ---------------------------
# Load spaCy Model
# ---------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ---------------------------
# Text Cleaning (SAME AS COLAB)
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(tokens)

# ---------------------------
# Load Data & Model
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_reviews.csv")

@st.cache_resource
def load_model():
    return load("best_sentiment_model.joblib")

df = load_data()
model = load_model()

label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "EDA Dashboard",
        "Sentiment Prediction"
    ]
)

# ===========================
# PAGE 1: Project Overview
# ===========================
if menu == "Project Overview":
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    Analyze ChatGPT user reviews and classify them into  
    **Positive, Neutral, or Negative** sentiments.

    **Tech Stack:**  
    - Python, NLP (spaCy)  
    - TF-IDF + SMOTE + ML Models  
    - Streamlit for deployment  

    **Business Impact:**  
    - Understand customer satisfaction  
    - Identify complaints  
    - Improve product experience  
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

# ===========================
# PAGE 2: EDA Dashboard
# ===========================
elif menu == "EDA Dashboard":
    st.header("📊 Exploratory Data Analysis")

    # --- Sentiment Distribution ---
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = df["sentiment"].map(label_mapping).value_counts()

    fig, ax = plt.subplots()
    sns.barplot(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        ax=ax
    )
    ax.set_ylabel("Number of Reviews")
    ax.set_xlabel("Sentiment")
    st.pyplot(fig)

    # --- Rating Distribution ---
    st.subheader("Distribution of Review Ratings")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=df, ax=ax)
    st.pyplot(fig)

    # --- Platform Comparison ---
    st.subheader("Average Rating by Platform")
    platform_avg = df.groupby("platform")["rating"].mean().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(
        data=platform_avg,
        x="platform",
        y="rating",
        ax=ax
    )
    ax.set_ylim(0, 5)
    st.pyplot(fig)

    # --- Word Cloud ---
    st.subheader("Most Common Words in Negative Reviews")
    negative_reviews = df[df["sentiment"] == 0]["clean_review"]

    if len(negative_reviews) > 0:
        negative_text = " ".join(negative_reviews)
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(negative_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No negative reviews available for WordCloud.")

# ===========================
# PAGE 3: Sentiment Prediction
# ===========================
elif menu == "Sentiment Prediction":
    st.header("🧠 Sentiment Prediction")

    user_input = st.text_area(
        "Enter a review text:",
        height=150,
        placeholder="Type or paste a ChatGPT review here..."
    )

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned_input = clean_text(user_input)
            prediction = model.predict([cleaned_input])[0]
            sentiment = label_mapping[prediction]
            st.success(f"Predicted Sentiment: **{sentiment}**")


    st.markdown("---")
    st.header("📌 Key Sentiment Analysis Questions & Insights")

    # 1️⃣ Overall sentiment
    st.subheader("1. Overall Sentiment of User Reviews")
    sentiment_counts = df["sentiment"].map(label_mapping).value_counts()
    st.bar_chart(sentiment_counts)

    # 2️⃣ Sentiment vs Rating
    st.subheader("2. Sentiment Variation by Rating")
    sentiment_rating = df.groupby(["rating", "sentiment"]).size().unstack().fillna(0)
    sentiment_rating.columns = sentiment_rating.columns.map(label_mapping)
    st.bar_chart(sentiment_rating)

    # 3️⃣ Keywords by Sentiment
    st.subheader("3. Keywords Associated with Each Sentiment")
    col1, col2, col3 = st.columns(3)

    for col, label, title in zip(
        [col1, col2, col3],
        [0, 1, 2],
        ["Negative", "Neutral", "Positive"]
    ):
        text = " ".join(df[df["sentiment"] == label]["clean_review"])
        wc = WordCloud(width=300, height=300, background_color="white").generate(text)
        with col:
            st.markdown(f"**{title} Reviews**")
            st.image(wc.to_array())

    # 4️⃣ Sentiment over time
    st.subheader("4. Sentiment Trend Over Time")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    time_sentiment = df.groupby([df["date"].dt.to_period("M"), "sentiment"]).size().unstack().fillna(0)
    time_sentiment.columns = time_sentiment.columns.map(label_mapping)
    st.line_chart(time_sentiment)

    # 5️⃣ Verified vs Non-Verified
    st.subheader("5. Verified vs Non-Verified Sentiment")
    verified_sent = df.groupby(["verified_purchase", "sentiment"]).size().unstack().fillna(0)
    verified_sent.columns = verified_sent.columns.map(label_mapping)
    st.bar_chart(verified_sent)

    # 6️⃣ Review Length vs Sentiment
    st.subheader("6. Review Length vs Sentiment")
    df["review_length"] = df["review"].astype(str).apply(len)
    length_sent = df.groupby("sentiment")["review_length"].mean()
    length_sent.index = length_sent.index.map(label_mapping)
    st.bar_chart(length_sent)

    # 7️⃣ Location vs Sentiment
    st.subheader("7. Location-wise Sentiment")
    top_locations = df["location"].value_counts().head(5).index
    loc_sent = df[df["location"].isin(top_locations)].groupby(
        ["location", "sentiment"]
    ).size().unstack().fillna(0)
    loc_sent.columns = loc_sent.columns.map(label_mapping)
    st.bar_chart(loc_sent)

    # 8️⃣ Platform vs Sentiment
    st.subheader("8. Platform-wise Sentiment")
    plat_sent = df.groupby(["platform", "sentiment"]).size().unstack().fillna(0)
    plat_sent.columns = plat_sent.columns.map(label_mapping)
    st.bar_chart(plat_sent)

    # 9️⃣ Version vs Sentiment
    st.subheader("9. Sentiment by ChatGPT Version")
    df["major_version"] = df["version"].astype(str).str.extract(r"(\d+\.\d+)")
    ver_sent = df.groupby(["major_version", "sentiment"]).size().unstack().fillna(0)
    ver_sent.columns = ver_sent.columns.map(label_mapping)
    st.bar_chart(ver_sent.head(5))

    # 🔟 Negative feedback themes
    st.subheader("10. Common Negative Feedback Themes")
    negative_text = " ".join(df[df["sentiment"] == 0]["clean_review"])
    neg_wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
    st.image(neg_wc.to_array())
