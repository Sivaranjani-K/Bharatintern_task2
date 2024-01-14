import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pandas as pd

ratings = pd.read_csv('d:/ML/ratings.csv')
movies = pd.read_csv('d:/ML/movies.csv')

movie_ratings = pd.merge(ratings, movies, on='movieId')

user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')


user_movie_matrix = user_movie_matrix.fillna(0)

movie_similarity = cosine_similarity(user_movie_matrix.T)

movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def get_movie_recommendations(movie_title, top_n=5):
    similar_scores = movie_similarity_df[movie_title]
    similar_movies = similar_scores.sort_values(ascending=False).index[1:top_n+1]
    return similar_movies

input = input("Enter a movie with year (i.e Toy Story (1995)): ")

if input in movie_similarity_df.columns:
    recommendations = get_movie_recommendations(input)
    print(f"Recommendations for {input}: ")
    print(recommendations)
else:
    print(f"Movie '{input}' not found in the dataset.")