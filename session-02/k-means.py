from collections import defaultdict
import numpy as np
import random
import os

cwd = os.getcwd()

class Member:
    def __init__(self, rd, label=None, doc_id=None):
        self._rd = rd
        self._label = label
        self._doc_id = doc_id
    
class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    
    def reset_members(self):
        self._members = []
    
    def add_members(self, member):
        self._members.append(member)

class Kmeans:
    # self._num_clusters: int: number of clusters
    # self._clusters: list: list of all Clusters (class)
    # self._centroids: list: list of all centroids
    # self._S float: overall clustering error
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._centroids = [] # list of centroids
        self._S = 0 # overall similarity
    
    # self._data: list: all data/members. Member._rd: 1D np array
    # self._label_count: dict: number of files per news group
    def load_data(self, data_path):
        def sparse_to_dense(sparse_rd, vocab_size):
            rd = [0 for _ in range(vocab_size)]
            # split into word id & tf-idf
            for tf_idf_with_word_id in sparse_rd.split():
                word_id = int(tf_idf_with_word_id.split(':')[0])
                tf_idf = float(tf_idf_with_word_id.split(':')[1])
                rd[word_id] = tf_idf
            # export txt into a vector where each element is the tf-idf value of a word in the corpus' entire vocab set. vector ~ document
            return np.array(rd)
        
        with open(data_path) as fp:
            # corpus, each file with its label + doc_id + tf-idf (sparse) vector. vector ~ document
            corpus_rd = fp.read().splitlines() 
        with open(cwd + '\\words-idfs.txt') as fp:
            # vocab of entire corpus
            vocab_size = len(fp.read().splitlines()) 
        
        # include data points corresponding to documents: rd (of doc) + label (of doc) + doc_id
        self._data = []
        # label = news group
        self._label_count = defaultdict(int)

        for doc_rd in corpus_rd:
            (label, doc_id) = int(doc_rd.split('<fff>')[0]), int(doc_rd.split('<fff>')[1])
            self._label_count[label] += 1
            rd = sparse_to_dense(doc_rd.split('<fff>')[2], vocab_size)

            # all data
            self._data.append(Member(rd, label, doc_id))

    def random_init(self, seed_value):
        random.seed(seed_value)
        data_shuffled = random.choices(self._data, k=self._num_clusters)
        for i in range(self._num_clusters):
            centroid = data_shuffled[i]._rd
            self._clusters[i]._centroid = centroid
            self._centroids.append(centroid)
        
    def compute_similarity(self, member, centroid):
        return sum(member._rd * centroid) / np.sqrt(np.sum(member._rd ** 2) * np.sum(centroid ** 2))

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        # add data point to cluster._members (list)
        best_fit_cluster.add_members(member)
        return max_similarity
    
    def update_centroid_of(self, cluster):
        aver_rd = np.mean([member._rd for member in cluster._members], axis=0)
        sum_squares = np.sum(aver_rd ** 2)
        # normalize average rd
        new_centroid = np.array([num / np.sqrt(sum_squares) for num in aver_rd])

        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria

        if criterion == 'max_iters':
            if self._iteration  >= threshold:
                return True
            else:
                return False
        
        elif criterion == 'centroid':
            # after calculating centroids, before re-positioning centroid
            centroids_new = [list(cluster._centroid) for cluster in self._clusters]
            centroids_changes = [centroid for centroid in centroids_new if centroid not in self._centroids]
            self._centroids = centroids_new
            
            if len(centroids_changes) <= threshold:
                return True
            else:
                return False
        
        elif criterion == 'similarity':
            # while running algorithm, keep separated previous clustering error (S) and new clustering error (new_S)
            clustering_error_change = self._new_S - self._S
            self._S = self._new_S
            if clustering_error_change <= threshold:
                return True
            else:
                return False

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)

        # continually update clusters and move centroids until convergence
        self._iteration = 0
        while True:
            # reset clusters, retain only centroids
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                # put member into cluster so that similarity is maximized, and return said similarity
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            
            self._iteration += 1
            if self.stopping_condition(criterion, threshold) == True:
                break

Kmeans_3 = Kmeans(3)
Kmeans_3.load_data(cwd + '\\data-tf-idf.txt')
Kmeans_3.run(1, 'max_iters', 10)
print(Kmeans_3._clusters)