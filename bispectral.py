import pickle
import numpy as np
from sklearn.cluster import SpectralCoclustering
from tqdm import tqdm
import json

# the main function for bi-spectral algorithm
def run_bispectral(input_data, min_user = 10, k = 100):

    # get mapping from user / hashtag name to idx, as well as from idx to user / hashtag name
    userid_to_idx = {}
    hashtag_to_idx = {}
    idx_to_userid = {}
    idx_to_hashtag = {}
    userid_to_username = {}
    for data_key in tqdm(input_data):
        userid = data_key[0]
        username = data_key[1]
        hashtag = data_key[2]
        userid_to_username[userid] = username
        cur_userid = len(userid_to_idx)
        cur_hashtagid = len(hashtag_to_idx)
        if userid not in userid_to_idx:
            userid_to_idx[userid] = cur_userid
            idx_to_userid[cur_userid] = userid
        if hashtag not in hashtag_to_idx:
            hashtag_to_idx[hashtag] = cur_hashtagid
            idx_to_hashtag[cur_hashtagid] = hashtag

    num_users = len(userid_to_idx)
    num_hashtags = len(hashtag_to_idx)


    # build adjacency matrices
    A = np.zeros((num_users, num_hashtags), dtype=np.uint16)
    S = np.zeros((num_users, num_hashtags), dtype=np.uint16)
    for data_key in tqdm(input_data):
        A[userid_to_idx[data_key[0]], hashtag_to_idx[data_key[2]]] = input_data[data_key]
        S[userid_to_idx[data_key[0]], hashtag_to_idx[data_key[2]]] = 1

    hashtag_cond = (S.sum(axis=0) >= min_user)
    hashtag_idx_mapping = get_idx_mapping(hashtag_cond) # generate mapping from old idx to new idx
    A = A[:, hashtag_cond]

    userid_cond = (A.sum(axis=1) > 0)
    userid_idx_mapping = get_idx_mapping(userid_cond)
    A = A[userid_cond, :]
    del S

    # run the bi-spectral clustering algorithm
    clustering = SpectralCoclustering(n_clusters=k, random_state=0).fit(A)
    cluster_row_labels = clustering.row_labels_
    cluster_col_labels = clustering.column_labels_

    # extract results from the bi-spectral clustering algorithm
    clustering_results = {}
    # initilize result dicts for k clusters
    for i in range(k):
        clustering_results[i] = {"users":[], "hashtags":[]}
    for idx, label in enumerate(list(cluster_row_labels)):
        label = int(label)
        userid = idx_to_userid[userid_idx_mapping[idx]]
        username = userid_to_username[userid]
        user_weight = int(sum(A[idx, :])) # user weight is the total weight of edges connected to the user
        clustering_results[label]["users"].append({"username": username, "weight": user_weight})

    for idx, label in enumerate(list(cluster_col_labels)):
        label = int(label)
        hashtagidx = hashtag_idx_mapping[idx]
        hashtag = idx_to_hashtag[hashtagidx]
        hashtag_weight = int(sum(A[:, idx]))
        clustering_results[label]["hashtags"].append({"hashtag": hashtag, "weight": hashtag_weight})
    
    json.dump(clustering_results, open("clustering_results.json", "w"))


    # in second pass clustering, we re-run the clustering algorithm on the biggest cluster from the first run 
    second_pass_row_cond = (cluster_row_labels == 96)
    second_pass_col_cond = (cluster_col_labels == 96)
    second_pass_userid_idx_mapping = get_idx_mapping(second_pass_row_cond)
    second_pass_hashtag_idx_mapping = get_idx_mapping(second_pass_col_cond)

    A = A[second_pass_row_cond, :]
    A = A[:, second_pass_col_cond]

    second_pass_clustering = SpectralCoclustering(n_clusters=k, random_state=0).fit(A)
    second_pass_cluster_row_labels = second_pass_clustering.row_labels_
    second_pass_cluster_col_labels = second_pass_clustering.column_labels_

    # extract the results from second-pass clustering
    second_pass_clustering_results = {}
    for i in range(k):
        second_pass_clustering_results[i] = {"users":[], "hashtags":[]}
    for idx, label in enumerate(list(second_pass_cluster_row_labels)):
        label = int(label)
        userid = idx_to_userid[userid_idx_mapping[second_pass_userid_idx_mapping[idx]]]
        username = userid_to_username[userid]
        user_weight = int(sum(A[idx, :]))
        second_pass_clustering_results[label]["users"].append({"username": username, "weight": user_weight})

    for idx, label in enumerate(list(second_pass_cluster_col_labels)):
        label = int(label)
        hashtagidx = hashtag_idx_mapping[second_pass_hashtag_idx_mapping[idx]]
        hashtag = idx_to_hashtag[hashtagidx]
        hashtag_weight = int(sum(A[:, idx]))
        second_pass_clustering_results[label]["hashtags"].append({"hashtag": hashtag, "weight": hashtag_weight})

    json.dump(second_pass_clustering_results, open("second_pass_clustering_results.json", "w"))


# generate an index mapping from new idx to old idx in the adjacency matrix
# useful since we use condition to filter out certain row and columns
def get_idx_mapping(condition):
    idx_mapping = {}
    cur_new_idx = 0
    for old_idx, c in enumerate(list(condition)):
        if c == True:
            idx_mapping[cur_new_idx] = old_idx
            cur_new_idx += 1

    return idx_mapping


# main method
def main():
    data_file = "./user_hashtag_weight.p"
    input_data = pickle.load(open(data_file, "rb"))
    run_bispectral(input_data)


if __name__ == "__main__":
    main()