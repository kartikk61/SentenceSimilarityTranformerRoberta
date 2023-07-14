from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords


# def load_data():
#     return SentenceTransformer("kartikkitukale61/RobertaSentenceSimilarityKartik")

model = SentenceTransformer("kartikkitukale61/RobertaSentenceSimilarityKartik")
# Load the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Convert tokens back to string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def calculate_similarity(sentence1, sentence2):
    # Preprocess input sentences
    preprocessed_sentence1 = preprocess_text(sentence1)
    preprocessed_sentence2 = preprocess_text(sentence2)

    # Encode sentences using the SentenceTransformer model
    sentence1_embeddings = model.encode([preprocessed_sentence1], convert_to_tensor=True)
    sentence2_embeddings = model.encode([preprocessed_sentence2], convert_to_tensor=True)

    # Calculate cosine similarity between the sentence embeddings
    similarity_score = util.pytorch_cos_sim(sentence1_embeddings, sentence2_embeddings)[0][0]

    return similarity_score.item()

# Streamlit app
def main():
    st.title("Sentence Similarity")
    st.write("Enter two sentences and get the similarity score.")

    # Input sentences
    sentence1 = st.text_input("Sentence 1")
    sentence2 = st.text_input("Sentence 2")

    if st.button("Calculate Similarity"):
        # Calculate the similarity score
        similarity_score = calculate_similarity(sentence1, sentence2)

        # Display the similarity score
        st.write("Similarity Score:", similarity_score)

if __name__ == "__main__":
    main()
