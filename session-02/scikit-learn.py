import numpy as np

def load_data(data_path):
        def sparse_to_dense(sparse_rd, vocab_size): # returns rd of one doc, as a list
            rd = [0 for _ in range(vocab_size)]
            # split into word id & tf-idf
            for tf_idf_with_word_id in sparse_rd.split():
                word_id = int(tf_idf_with_word_id.split(':')[0])
                tf_idf = float(tf_idf_with_word_id.split(':')[1])
                rd[word_id] = tf_idf
            # export txt into a vector where each element is the tf-idf value of a word in the corpus' entire vocab set. vector ~ document
            return rd
        
        with open(data_path) as fp:
            # corpus, each file with its label + doc_id + tf-idf (sparse) vector. vector ~ document
            corpus_rd = fp.read().splitlines() 
        with open('words-idfs.txt') as fp:
            # vocab of entire corpus
            vocab_size = len(fp.read().splitlines()) 

        labels = list()
        data = list()
        for doc_rd in corpus_rd:
            label = int(doc_rd.split('<fff>')[0])
            labels.append(label)
            rd = sparse_to_dense(doc_rd.split('<fff>')[2], vocab_size)
            data.append(rd)
        
        return np.array(data), np.array(labels)

def clustering_with_KMeans():
    
    data, labels = load_data('data-full-tf-idf.txt')
    # use csr_matrix to create a sparse matrix with efficient row slicing
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print('=========')
    kmeans = KMeans(n_clusters = 20, 
                    init = 'random',
                    n_init = 5, # number of times that kmeans runs with differently initialized centroids
                    tol = 1e-3, # threshold for acceptable minimum error decrease
                    random_state = 2023 # set to get deterministic results (seed)
                    ).fit(X)
    
    kmeans_labels = kmeans.labels_
    print(labels[:25])
    print(kmeans_labels[:25])

def classifying_with_linear_SVMs():
    def compute_accuracy(predicted_y, expected_y):
        matches = np.equal(predicted_y, expected_y)
        accuracy = np.sum(matches.astype(float)) / expected_y.size
        return accuracy

    train_X, train_y = load_data('data-train-tf-idf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C = 10, # regularization
        tol = 0.001, # tolerance for stopping criteria
        verbose = True # whether prints out logs or not
    )
    classifier.fit(train_X, train_y)

    test_X, test_y = load_data('data-test-tf-idf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print('Accuracy:', accuracy)

def classifying_with_kernel_SVMs():
    def compute_accuracy(predicted_y, expected_y):
        matches = np.equal(predicted_y, expected_y)
        accuracy = np.sum(matches.astype(float)) / expected_y.size
        return accuracy

    train_X, train_y = load_data('data-train-tf-idf.txt')
    from sklearn.svm import SVC
    classifier = SVC(
        C = 50,
        kernel = 'rbf',
        gamma = 0.1,
        tol = 0.001,
        verbose = True
    )
    classifier.fit(train_X, train_y)

    test_X, test_y = load_data('data-test-tf-idf.txt')
    predicted_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
    print('Accuracy:', accuracy)

def main():
    #clustering_with_KMeans()
    #classifying_with_linear_SVMs()
    classifying_with_kernel_SVMs()

if __name__ == '__main__':
    main()