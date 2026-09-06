import os

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = os.environ.get("MODEL_NAME")


@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


st.title("🇫🇷 Informal French Similarity (FlauBERT)")
st.write("Compare two informal French sentences with a fine-tuned regression model.")

if not MODEL_NAME:
    st.error(
        "MODEL_NAME is required. Set it to a compatible fine-tuned "
        "Hugging Face model name or local checkpoint path."
    )
    st.stop()

tokenizer, model = load_model(MODEL_NAME)
sentence1 = st.text_input("Sentence 1", "wsh t'es ou ?")
sentence2 = st.text_input("Sentence 2", "tu es où ?")

if st.button("Compute similarity"):
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        score = model(**inputs).logits.squeeze().item()

    st.success(f"Similarity score: {score:.3f}")
