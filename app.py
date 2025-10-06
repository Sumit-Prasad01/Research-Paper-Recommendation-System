import streamlit as st 
import torch
import pickle
from sentence_transformers import  util



embeddings = pickle.load(open("artifacts/models/embeddings.pkl", 'rb'))
sentences = pickle.load(open("artifacts/models/sentences.pkl", 'rb'))
rec_model = pickle.load(open("artifacts/models/rec_model.pkl", 'rb'))



def recommendation(input_paper):
    # Encode input and compute cosine similarity
    input_embedding = rec_model.encode(input_paper)
    cosine_scores = util.cos_sim(embeddings, input_embedding).squeeze(1)
    
    # Sort all papers by similarity (descending)
    sorted_indices = torch.argsort(cosine_scores, descending=True)
    
    # Collect up to 5 unique papers
    papers_list = []
    seen = set()
    
    for idx in sorted_indices:
        title = sentences[idx.item()]
        if title not in seen:
            papers_list.append(title)
            seen.add(title)
        if len(papers_list) == 5:
            break
    
    return papers_list


st.title("Research Paper Recommendation System")
input_paper = st.text_input("Enter the paer title : ")

if st.button("recommend"):
    recommended_papers = recommendation(input_paper)
    st.subheader("Recomemnded Papers : ")
    st.write(recommended_papers)