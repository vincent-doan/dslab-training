import os, re
from collections import defaultdict
from numpy import log10, sqrt

def gather_20newsgroups_data():
    path = 'D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/'
    # add path to subfolders only, not existing files in path
    dirs = [path + dir_name + '/' for dir_name in os.listdir(path) if not os.path.isfile(path + dir_name)]
    # assign name train_dir and test_dir to "path to subfolders"
    (train_dir, test_dir) = (dirs[0],dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    
    # inside subfolder .../20news-bydate-train are even more sub-subfolders, in this case news groups
    list_newsgroups = [newsgroups for newsgroups in os.listdir(train_dir)]
    list_newsgroups.sort()

    with open('D:/Education/HUST/dslab-706/session-01/datasets/stop_words.txt') as f:
        # read from stop_words.txt file to get list of stop words
        stop_words = f.read().splitlines()

    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()

    def collect_data_from(parent_dir, list_newsgroups):
        data = list()
        for group_id, newsgroup in enumerate(list_newsgroups):
            label = group_id
            # path to sub-subfolders, in this case the news groups
            dir_path = parent_dir + '/' + newsgroup + '/'
            # add (filename, filepath) for files only
            files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path + filename)]
            files.sort()
            
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    # remove stop words, then stem remaining words
                    # re.split('\W+', text) splits the file into blocks of text, skipping over blank lines etc.
                    words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
                    # combine remaining words
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    # add label of news group, file name of a file, and its content on each iteration
                    # do this for files in a news group, for all news group 
                    data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        return data
    
    #list of (news group's index + file name of file + content)
    train_data = collect_data_from(train_dir, list_newsgroups)
    test_data = collect_data_from(test_dir, list_newsgroups)

    full_data = train_data + test_data
    # export data into a txt file, each news file corresponding to a line
    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))

def generate_vocabulary(data_path):
    def compute_idf(document_freq, corpus_size):
        assert document_freq > 0
        return log10(corpus_size / document_freq)

    with open(data_path) as f:
        # each line correspond to a document, the entire file at data_path is the corpus
        corpus = f.read().splitlines()
    
    corpus_size = len(corpus)
    
    # default dictionary to have 0 frequency for words not in dictionary
    doc_count = defaultdict(int)
    for document in corpus:
        content = document.split('<fff>')[-1]
        # get the words in a document, regardless of term frequency
        words = list(set(content.split()))
        # doc_count keeps track of the number of documents containing a word. key = word, value = document frequency
        for word in words:
            doc_count[word] += 1

    # get list of idfs of words, ignore words that appear in too few documents, ignore numbers    
    words_idfs = [(word, compute_idf(document_freq, corpus_size)) for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq > 10 and not word.isdigit()]
    # sort by document frequency, in decreasing order
    words_idfs.sort(key = lambda t: t[1], reverse = True)
    print("Vocabulary size:",len(words_idfs))
    
    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/words-idfs.txt', 'w') as f:
        # export to a txt file, with each line being a word and its idf (inverse document frequency)
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))

def get_tf_idf(data_path):
    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/words-idfs.txt') as f:
        # return list or tuples of the form (word, inverse-document-frequency of word)
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1])) for line in f.read().splitlines()]
        # dictionary where key is the word and value is associated index - for simpler look-up - each word in total dictionary gets an ID
        word_IDs = dict([(word, index) for index, (word, idf) in enumerate(words_idfs)])
        # dictionary where key is the word and value is associated idf - for simpler look-up
        idfs = dict(words_idfs)
    
    with open(data_path) as f:
        # list with each element being a tuple (index of news group, file name, content)
        corpus = [(int(line.split('<fff>')[0]), int(line.split('<fff>')[1]), line.split('<fff>')[2]) for line in f.read().splitlines()]
        
    data_tf_idf = list()
    for document in corpus:
        label, doc_id,  text = document
        # a list that includes all words in a document if the word has idf. The word can be repeated in a list, to document term frequency
        words = [word for word in text.split() if word in idfs]
        # create an non-repeating iterable of words in a document
        word_set = list(set(words))
        # maximum term frequency for a document
        max_term_freq = max([words.count(word) for word in word_set])

        # document_tf_idfs : a vector with tf-idf value of words in the entire dictionary, with respect to a particular document, list of tuples
        document_tf_idfs = list()
        sum_squares = 0
        for word in word_set:
            # calculate term frequency for a particular word in a document, then look up idf to compute tf-idf
            term_freq = words.count(word)
            tf_idf = term_freq * idfs[word] / max_term_freq
            # document_tf_idfs is a list of tuples (word ID, tf-idf of word). note this is word ID instead of word
            document_tf_idfs.append((word_IDs[word], tf_idf))
            sum_squares += tf_idf ** 2
        
        document_tf_idfs_normalized = [str(word_index) + ':' + str(tf_idf / sqrt(sum_squares)) for word_index, tf_idf in document_tf_idfs]

        sparse_rep = ' '.join(document_tf_idfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open('D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/data-tf-idf.txt', 'w') as f:
        # export to a txt file, with each line being a document: news group, file name, tf-idf vector
        f.write('\n'.join(str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep for label, doc_id, sparse_rep in data_tf_idf))

def main():
    #generate_vocabulary("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-train-processed.txt")
    #generate_vocabulary("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-test-processed.txt")
    generate_vocabulary("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-full-processed.txt")

    #get_tf_idf("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-train-processed.txt")
    #get_tf_idf("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-test-processed.txt")
    get_tf_idf("D:/Education/HUST/dslab-706/session-01/datasets/20news-bydate/20news-bydate-full-processed.txt")

if __name__ == '__main__':
    main()