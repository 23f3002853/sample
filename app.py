import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pretrained models (DeBERTa + RoBERTa)
tok_d = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
deberta = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small", num_labels=5
)

tok_r = AutoTokenizer.from_pretrained("roberta-base")
roberta = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=5
)

options = ["A", "B", "C", "D", "E"]

# Functions for probability extraction and ensembling
def get_probs(model, tokenizer, prompt, opts):
    inputs = tokenizer([prompt + " " + opt for opt in opts],
                       padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()[0]

def ensemble_probs(p_d, p_r, w1=0.7, w2=0.3):
    return w1 * p_d + w2 * p_r

def top3_string(probs):
    top3 = probs.argsort()[-3:][::-1]
    return " ".join([options[i] for i in top3])

# Streamlit UI
st.title("Smart MCQ Solver – DL & GenAI Project")
prompt = st.text_input("Enter your MCQ prompt/question:")

if prompt:
    p_d = get_probs(deberta, tok_d, prompt, options)
    p_r = get_probs(roberta, tok_r, prompt, options)
    probs = ensemble_probs(p_d, p_r)
    st.write("Top‑3 Predictions:", top3_string(probs))
