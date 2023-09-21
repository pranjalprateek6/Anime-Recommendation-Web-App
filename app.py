import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
import re

# Load the IMDb anime dataset and filter it
data = pd.read_csv("imdb_anime.csv")
data = data[pd.to_numeric(data['User Rating'], errors='coerce').notna()]
data['Genres'] = data['Genre'].str.split(', ')
data['User Rating'] = data['User Rating'].astype(float)
data['Gross'] = data['Gross'].str.replace(',', '').astype(float)

# Clean the 'Year' column by extracting valid numeric values
data['Year'] = data['Year'].apply(lambda x: re.findall(r'\d+', str(x)))
data['Year'] = data['Year'].apply(lambda x: int(x[0]) if x else None)

# Create TF-IDF matrix as a CSR sparse matrix
tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[a-zA-Z0-9]+', lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Genre'].apply(lambda x: ' '.join(x)))
tfidf_matrix = csr_matrix(tfidf_matrix)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get anime recommendations
def get_recommendations(title):
    idx = data[data['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar anime
    anime_indices = [i[0] for i in sim_scores]
    return data[['Title', 'Genre', 'User Rating', 'Gross']].iloc[anime_indices]

# Streamlit app
st.title("Anime Recommendation App")
st.write("Enter the name of an anime you like, and we'll recommend similar anime.")

# User input
user_input = st.text_input("Enter the name of an anime:", "Death Note")

if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_input)
    st.subheader("Recommended Anime:")
    st.write(recommendations)
