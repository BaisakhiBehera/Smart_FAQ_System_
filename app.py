import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🤖 Smart FAQ Auto-Responder")

# Step 1: Upload FAQ file or use default
uploaded_file = st.file_uploader("Upload FAQ file (CSV with 'question' and 'answer' columns)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("faq_data.csv")


# Step 2: Generate embeddings for FAQ questions
faq_questions = df['question'].tolist()
faq_embeddings = model.encode(faq_questions)

# Step 3: User query input
user_query = st.text_input("Ask your question:")

if user_query:
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, faq_embeddings)
    top_idx = np.argmax(similarities)
    
    matched_question = df.iloc[top_idx]['question']
    matched_answer = df.iloc[top_idx]['answer']
    confidence = float(similarities[0][top_idx]) * 100

    threshold = 60  # 60%

    if confidence >= threshold:
      #  st.markdown(f"**You:** {user_query}")
        st.markdown(f"**Bot:** {matched_answer}")
        st.markdown(f"**Matched FAQ:** _{matched_question}_")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
    else:
        st.markdown("**Bot Answer:** 👋 Hey, sorry I couldn't find a good answer for your question.")
        st.markdown("Please reach out to **faqsupport@xyz.com** and our team will get back to you shortly. 💌")
        st.markdown(f"**Confidence:** {confidence:.2f}% (too low for a match)")