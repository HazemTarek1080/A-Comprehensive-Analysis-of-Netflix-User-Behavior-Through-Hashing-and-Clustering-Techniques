import pandas as pd
import numpy as np

def min_hash_signatures(matrix, perms, num_buckets):
    """
    Calculate min-hash signatures for users based on their likes in a binary matrix.

    Parameters:
    - matrix (numpy.ndarray): Binary matrix representing user likes (genres).
    - perms (int): Number of permutations for generating min-hash signatures.
    - num_buckets (int): Number of buckets to hash signatures.

    Returns:
    - signature_matrix (numpy.ndarray): Matrix containing min-hash signatures.
    - buckets (dict): Dictionary mapping bucket index to a list of user indices.
    """
    num_users = matrix.shape[1]

    # Initialize the signature matrix with infinity values
    signature_matrix = np.full((perms, num_users), np.inf)

    # Generate random hash functions
    hash_functions = [np.random.permutation(num_users) for _ in range(perms)]

    # Initialize an empty dictionary to store buckets
    buckets = {i: [] for i in range(num_buckets)}

    # Iterate through each row (genre) in the binary matrix
    for genre_index in range(matrix.shape[0]):
        # Check if the genre is liked by each user
        genre_likes = matrix[genre_index, :]

        # Update the min-hash signatures
        for hash_index, hash_function in enumerate(hash_functions):
            # Find users who like the genre
            users_in_genre = np.where(genre_likes == 1)[0]

            # Update the signature if the user likes the genre
            signature_matrix[hash_index, users_in_genre] = np.minimum(
                signature_matrix[hash_index, users_in_genre],
                hash_function[genre_index]
            )

        # Map the min-hash signature to a bucket
        for user_index in range(num_users):
            bucket_index = hash(tuple(signature_matrix[:, user_index])) % num_buckets
            buckets[bucket_index].append(user_index)  # Add the user to the corresponding bucket

    return signature_matrix, buckets


def find_similar_users(target_user_id, unique_users, signatures):
    """
    Find the two most similar users to a target user based on Jaccard similarity.

    Parameters:
    - target_user_id (int): ID of the target user.
    - unique_users (list): List of unique user IDs.
    - signatures (numpy.ndarray): Matrix containing min-hash signatures.

    Returns:
    - most_similar_users (list): List of tuples containing user IDs and their Jaccard similarity.
    """
    # Index of the target user
    target_index = unique_users.index(target_user_id)

    # Calculate Jaccard similarity with all other users
    similarities = []
    for user_index, other_user_id in enumerate(unique_users):

        # Skip the target user
        if user_index != target_index:
            # Calculate Jaccard similarity
            intersection = np.sum(np.minimum(signatures[:, target_index], signatures[:, user_index]))
            union = np.sum(np.maximum(signatures[:, target_index], signatures[:, user_index]))
            similarity = intersection / union

            # Append the similarity and user ID to the list
            similarities.append((other_user_id, similarity))

    # Sort the list by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the two most similar users
    most_similar_users = similarities[:2]

    return most_similar_users


def get_top5_movies(user_id, df, user_label):
    """
    Get the top 5 movies clicked by a user.

    Parameters:
    - user_id (int): ID of the target user.
    - df (pandas.DataFrame): DataFrame containing user clicks data.
    - user_label (str): Label for the target user.

    Returns:
    - top_5_movies_titles (pandas.DataFrame): DataFrame with the top 5 movies titles, movie IDs, clicks, and user label.
    """
    # Filter the DataFrame to the target user
    user_df = df[df['user_id'] == user_id]

    # Get the top 5 movies that appeared most in the user clicks
    top_5_movies = user_df['movie_id'].value_counts()[:5]

    # Get the top 5 movies titles
    top_5_movies_titles = df[df['movie_id'].isin(top_5_movies.index)][['title', 'movie_id']]

    # Remove duplicates
    top_5_movies_titles.drop_duplicates(inplace=True)

    # Remove the movies with title 'NOT AVAILABLE'
    top_5_movies_titles = top_5_movies_titles[top_5_movies_titles['title'] != 'NOT AVAILABLE']

    # Add the number of clicks as a column
    top_5_movies_titles['clicks'] = top_5_movies.values

    # Remove the index column
    top_5_movies_titles.reset_index(drop=True, inplace=True)

    # Add the user column with the value of user_label for all rows
    top_5_movies_titles['user'] = user_label

    return top_5_movies_titles


def recommend_movies(user_A_movies, user_B_movies):
    """
    Recommend movies based on common clicks between two users.

    Parameters:
    - user_A_movies (pandas.DataFrame): DataFrame of movies clicked by user A.
    - user_B_movies (pandas.DataFrame): DataFrame of movies clicked by user B.

    Returns:
    - recommend_movies (list): List of recommended movie titles.
    """
    # Merge user_A_movies and user_B_movies to find common movies
    common_movies_df = pd.merge(user_A_movies, user_B_movies, on="movie_id", how="inner")

    # If there are common movies, recommend based on total clicks
    if not common_movies_df.empty:
        # Calculate total clicks for each common movie
        common_movies_df["total_clicks"] = common_movies_df["clicks_x"] + common_movies_df["clicks_y"]

        # Sort movies based on total clicks (descending order)
        sorted_movies_df = common_movies_df.sort_values(by="total_clicks", ascending=False)

        # Extract up to five recommended movies
        recommended_movies_df = sorted_movies_df.head(5)[["title_x", "movie_id", "total_clicks"]].rename(columns={"title_x": "title"})

    # Recommend most clicked movies from each user
    clicks_A = user_A_movies.sort_values(by="clicks", ascending=False).head(5)
    clicks_B = user_B_movies.sort_values(by="clicks", ascending=False).head(5)

    # Concatenate and deduplicate recommended movies
    recommended_movies_df = pd.concat([clicks_A, clicks_B]).drop_duplicates(subset="movie_id").head(5)

    recommend_movies = list(recommended_movies_df["title"])

    return recommend_movies
