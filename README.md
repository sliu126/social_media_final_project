# CS7290: Final Project

The main data file is on the discovery cluster: /home/liu.shij/social_media_final/user_hashtag_weight.p. It is too big (758M) to fit in a github repo.

The data is pickled as a Python dictionary. The keys are (userid, username, hashtag) tuples denoting an edge in the bipartite graph. An edge means that the user is mentioning the hashtag in the original tweets data. The value of the dictionary is the edge weight, indicating the number of times the user uses the hashtag.

Run "python bispectral.py" to generate results.