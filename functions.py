import pandas as pd
import numpy as np
import random


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


def extract_starting_centroids(rdd, k):
    #returns k samples extracted from our dataset(in rdd form) without replacement
    return rdd.takeSample(False, k)

def euclidean_squared_distance(x, y): #Computes the square of the euclidean distance between two points 
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return (np.linalg.norm(x-y))**2

# Function used to check the convergence of the Kmeans algorithm: checks if the centroids of two following iterations are the same 
checkconvergencefun = lambda l1, l2: sum(sum([el1 - el2 for (el1, el2) in zip(l1, l2)]))

def Kmeans(rdd_data, centroids, k, it):
    niter = it
    # MAP PHASE: assign to each point a key-value pair -> key: the label of the cluster it's assigned to; value: (point, 1) 
    rddmap = rdd_data.map(lambda point:(np.argmin([ euclidean_squared_distance(point, centroid) for centroid in centroids ]), (point, 1)))

    # REDUCE PHASE: for each key(cluster) we get (sum_{point in the cluster}, #{points in the clusters})
    rddreduce = rddmap.reduceByKey(lambda t1, t2: (np.add(t1[0], t2[0]), t1[1] + t2[1]) )

    # COMPUTE THE NEW CENTROIDS: new centroid = mean of the points in each cluster = sum_{point in the cluster}/#{points in the clusters} => for each point, 
    new_centroids = rddreduce.mapValues(lambda t: t[0]/t[1]).map(lambda t: t[1])
    new_centr_list = new_centroids.collect()

    # CHECKING CONVERGENCE 
    if checkconvergencefun([np.array(centroid) for centroid in centroids], new_centr_list) == 0:
        # Compute the clusters
        clusters = rddmap.groupByKey().sortByKey().mapValues(lambda iterable: [t[0] for t in list(iterable)]).collect()
        # List of keys corresponding to the cluster each point its assigned to (ordered by appearence of the points in the dataset)
        clusters_idx  = rddmap.map(lambda t: t[0]).collect()
        return clusters, clusters_idx, new_centr_list, niter
    else:
        # Update the number of iterations
        niter += 1
        # Iterative call 
        return Kmeans(rdd_data, new_centr_list, k, niter)

def get_centroids_Kmeans_pp(data, k):
    n = len(data)
    data = data.values
    
    # Initialize the list of centroids indeces by choosing the first one uniformly at random and saving it 
    centroids_indeces = [random.randint(0, n)]
    centroids = [np.array(data[centroids_indeces[-1], :])]

    # Compute the remaining k-1 centroids
    for _ in range(1, k):
        # Initialize list with squared distances from the nearest centeroids
        distances_nc = []

        for point in data:

            # Find the nearest centroid between the ones already computed and save its distance from the point 
            # (we can directly use the squared distance beacause is an increasing function in [0, +\infty) => we "save" computational cost])
            distances = np.array([euclidean_squared_distance(point, centroid) for centroid in centroids]).flatten()#compute sistances between point - centroids
            distances_nc.append(np.min(distances, axis =0)) #save the smallest distance

        # Choose the new centroid wrt their distance from the nearest centroid between the ones already computed
        centroids.append(random.choices(data, weights=distances_nc, k=1)[0])
        
    return centroids        
       

def kmeans_plusplus(datardd, datapddf, k):
    # Initialize the centroids for the first iteration
    centroids_pp = get_centroids_Kmeans_pp(datapddf, k)

    # Use the previous clustering (k-Means) algorithm to compute the clusters
    return Kmeans(datardd, centroids_pp, k, 0)