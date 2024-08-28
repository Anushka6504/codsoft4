import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Prepare Data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    'movie_id': [1, 2, 3, 1, 4, 2, 3, 4, 2, 4, 1, 3],
    'rating': [4, 5, 2, 5, 3, 3, 4, 2, 4, 5, 3, 4]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Step 2: Create the User-Item Matrix
user_movie_matrix = df.pivot(index='user_id', columns='movie_id', values='rating')
print("\nUser-Movie Matrix:")
print(user_movie_matrix)

# Step 3: Compute the Similarity Matrix
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
print("\nUser Similarity Matrix:")
print(user_similarity_df)

# Step 4: Make Recommendations
def recommend_movies(user_id, user_movie_matrix, user_similarity_df, num_recommendations=2):
    # Check if the user_id exists in the user_movie_matrix
    if user_id not in user_movie_matrix.index:
        return "User ID not found."
    
    # Get similar users, excluding the user itself
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Aggregate ratings of similar users and calculate the mean
    movie_ratings = user_movie_matrix.loc[similar_users].mean(axis=0)
    
    # Exclude movies already watched by the user
    user_watched_movies = user_movie_matrix.loc[user_id].dropna().index
    recommended_movies = movie_ratings.drop(user_watched_movies).sort_values(ascending=False).head(num_recommendations)
    
    return recommended_movies.index.tolist()

# Step 5: Get User Input and Test the System
try:
    user_id = int(input("\nEnter user ID for recommendations: "))
    recommended_movies = recommend_movies(user_id, user_movie_matrix, user_similarity_df)
    if isinstance(recommended_movies, list):
        print(f"\nRecommended movies for user {user_id}: {recommended_movies}")
    else:
        print(recommended_movies)
except ValueError:
    print("Please enter a valid integer for the user ID.")
