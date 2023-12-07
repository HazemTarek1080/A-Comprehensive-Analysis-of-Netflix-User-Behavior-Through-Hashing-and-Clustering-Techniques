# ADM-HW4: A Comprehensive Analysis of Netflix User Behavior Through Hashing and Clustering Techniques
This repository contains the source code, notebooks, and additional materials related to Assignment 4 of the "Algorithms for Data Mining" course (23/24) from the Data Science Master's program at Sapienza University of Rome. Below, you will find an overview of the contents of this repository.



## Description

In this assignment, Analysis is being done on Netflix users' data to gain a comprehensive understanding of user behavior within the platform, the strategy involves the application of hashing and clustering techniques. Hashing, a one-way transformation process, is utilized to protect sensitive user data or generate unique identifiers without compromising privacy. Meanwhile, clustering techniques such as k-means are employed to group users with similar behavioral patterns. By hashing user identifiers and clustering based on behavior, the platform can extract relevant information and highlights from user interactions. This approach allows for the analysis of usage patterns, identification of popular features, and understanding of specific user segments.

### <img src="/images/target.png" width="150" height="150" align="right" />


The goal is to carry out the following tasks:
- **Recommendation System:**
  - Implementation of a recommendation system for personalized content, products, and services based on user preferences.
  - Utilization of a custom Locality-Sensitive Hashing (LSH) algorithm to identify similar users and recommend highly-watched movies by those users.
  
- **Grouping Users (Clustering)**:

  - Feature engineering using user clicks data, creating features such as favorite genre, average click duration, time of day engagement, movie preference era, and     average daily time spent.
  - Consideration of normalization for scaling features and application of dimensionality reduction using Principal Component Analysis (PCA).
  - Implementation of K-means clustering algorithm from scratch in MapReduce, determining optimal clusters, and comparing with K-means++ results.
  - Characterization of clusters through pivot tables and analysis of relevant variables.
  
- **Density-Based Clustering:**
  - Exploration of Density-Based Clustering algorithms like OPTICS and DBSCAN on the same dataset used for K-means.
  - Analysis of results and comparison with centroid-based clustering.
    
- **Command Line Task:**

  - Utilization of command line tools to answer questions about Netflix data, including the most-watched title, average time between clicks, and user spending the most time on Netflix.
    
- **Algorithmic task:**
  - Explanation and demonstration of a recursive algorithm for Federico's exam score optimization.
  - Formal representation of the algorithm's time complexity using big-O notation.
  - Potential optimization of the algorithm to improve time complexity, with formal proof.


For a detailed understanding of the assignment requirements and problems, refer to this [link](https://github.com/Sapienza-University-Rome/ADM/tree/master/2023/Homework_4).



## Dataset
  The dataset focuses on the browsing behavior of Netflix users in the UK who willingly agreed to have their anonymized activity monitored. It specifically captures desktop and laptop activities, accounting for approximately 25% of global traffic. The data spans from January 2017 to June 2019 and exclusively documents instances when a user within the tracked panel in the UK clicked on a Netflix.com/watch URL for a movie. 

You can find the dataset used in this [link](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies).

<p align="center">
<img src="https://recoai.net/wp-content/uploads/2022/04/netflix.jpg" width = 600>
</p>

## Repo content

- **main.ipynb:** A comprehensive Jupyter notebook that presents a detailed analysis of the data. It includes a systematic breakdown of the analysis process, the resulting insights, and accompanying explanations to give a better understanding of the findings.
- **functions.py:** A Python script that includes the implementation of the custom functions that were used in the Jupyter Notebook.
- **commandline.sh:** A PowerShell script that carries out the Command Line Task.


## Usage
- Clone the repo using the command 👉 **git clone https://github.com/marinazanoni/ADM-HW4.git** in a bash script terminal.
- Open the Jupyter Notebook using your desired IDE and start following the instructions there in order to replicate the results.

## Collaborators
### <img src="/images/robot.png" width="150" height="150" align="right" />
- Hazem Aboulfotouh (aboulfotouh.2105193@studenti.uniroma1.it)
- Marina Zanoni (zanoni.1964213@studenti.uniroma1.it)
- Camilla cognome (Sapienza email)
- Ortal Hadad (ortal2824@gmail.com)

