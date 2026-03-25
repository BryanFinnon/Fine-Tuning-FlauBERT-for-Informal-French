import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model (replace with your fine-tuned model path if local)
MODEL_NAME = "flaubert/flaubert_base_cased"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🇫🇷 Informal French Similarity (FlauBERT)")

st.write("Compare two informal French sentences using fine-tuned embeddings.")

# Input
sentence1 = st.text_input("Sentence 1", "wsh t'es ou ?")
sentence2 = st.text_input("Sentence 2", "tu es où ?")

if st.button("Compute Similarity"):
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        output = model(**inputs)
        score = output.logits.squeeze().item()

    st.success(f"Similarity Score: {round(score, 3)}")

    if score > 3:
        st.write("🟢 Similar")
    else:
        st.write("🔴 Not Similar")
