# AI Echo: Your Smartest Conversational Partner

AI Echo is a sentiment analysis project that analyzes ChatGPT user reviews and classifies them into
Positive, Neutral, or Negative sentiments using NLP, Machine Learning, and Deep Learning techniques.

## Domain
Customer Experience & Business Analytics

## Features
- Text preprocessing using spaCy (lemmatization, stopword removal)
- Sentiment classification using ML models (Naive Bayes, Logistic Regression, Random Forest, AdaBoost)
- Deep Learning model using LSTM
- Evaluation using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC
- Interactive Streamlit dashboard
- Key sentiment analysis insights visualized in Streamlit

## Tech Stack
- Python
- NLP (spaCy)
- Machine Learning (Scikit-learn, TF-IDF, SMOTE)
- Deep Learning (Keras, LSTM)
- Streamlit

## How to Run
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Download spaCy model:
   python -m spacy download en_core_web_sm
4. Run the app:
   streamlit run main.py

## Dataset
ChatGPT user reviews dataset with ratings, reviews, platform, location, version, and verified status.

## Output
- Sentiment prediction (Positive / Neutral / Negative)
- Visual insights answering 10 key sentiment analysis questions
