import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, vocab_size, hidden_size, num_classes):
        self._vocab_size = vocab_size # input
        self._hidden_size = hidden_size # hidden
        self._num_classes = num_classes # output
    
    def build_graph(self):

        self._w_1 = tf.Variable(
            name = "weights_input_hidden",
            initial_value = tf.random.normal(shape = [self._vocab_size, self._hidden_size],
                                            stddev = 0.1,
                                            seed = 2022)
        )

        self._b_1 = tf.Variable(
            tf.zeros([1, self._hidden_size]),
            name='biases_input_hidden'
        )
        
        self._w_2 = tf.Variable(
            name = "weights_hidden_output",
            initial_value = tf.random.normal(shape = [self._hidden_size, self._num_classes],
                                            stddev = 0.1,
                                            seed = 2022)
        )
        
        self._b_2 = tf.Variable(
            tf.zeros([1, self._num_classes]),
            name='biases_hidden_output',
        )

        self._trainable_variables = [self._w_1, self._w_2, self._b_1, self._b_2]
    
    def forward(self, X):
        X_tf = tf.cast(X, dtype=tf.float32)
        hidden = tf.matmul(X_tf, self._w_1) + self._b_1
        hidden = tf.sigmoid(hidden)
        # logits = raw, unnormalized output of a model before transformed into a probability distribution
        logits = tf.matmul(hidden, self._w_2) + self._b_2
        return logits
    
    def predict(self, X):
        logits = self.forward(X)
        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)
        return predicted_labels
    
    def compute_loss(self, logits, real_Y):
        # convert a list of indices to represent each index with a binary vector of length qual to num_classes
        labels_one_hot = tf.one_hot(indices=real_Y, depth=self._num_classes, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss
    
    def train_step(self, X, y, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        with tf.GradientTape() as tape:
            logits = self.forward(X)
            current_loss = self.compute_loss(logits, y)
        grads = tape.gradient(current_loss, self._trainable_variables)
        optimizer.apply_gradients(zip(grads, self._trainable_variables))
        return current_loss
    
    def reset_parameters(self):
        self.build_graph()

class DataReader:
    def __init__(self, data_path, batch_size, vocab_size):
        # batch = subset of training data processed at once
        # epoch = a full pass through training dataset
        # iteration = one step of optimization algorithm
        self._batch_size = batch_size
        with open(data_path) as f:
            d_lines = f.read().splitlines()
        
        self._data = []
        self._labels = []
        
        for data_id, line in enumerate(d_lines):
            vector = [0 for _ in range(vocab_size)]
            features = line.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            all_words_with_tf_idf = features[2].split()
            for word in all_words_with_tf_idf:
                word_id, tf_idf = int(word.split(':')[0]), float(word.split(':')[1])
                vector[word_id] = tf_idf
            self._data.append(vector)
            self._labels.append(label)
        
        self._data = np.array(self._data)
        self._labels = np.array(self._labels)

        self._num_epoch = 0
        self._batch_id = 0

    def next_batch(self):
        start = self._batch_id * self._batch_size
        end = start + self._batch_size
        self._batch_id += 1

        if end + self._batch_size > len(self._data):
            # complete 1 epoch, final batch not enough, then stop at 'end'
            end = len(self._data)
            self._num_epoch += 1
            self._batch_id = 0

            # shuffle dataset
            indices = np.arange(self._data.shape[0])
            np.random.seed(2023)
            np.random.shuffle(indices)
            self._data, self._labels = self._data[indices], self._labels[indices]
        
        return self._data[start:end], self._labels[start:end]
    
    def reset(self):
        self._batch_id = 0
        self._num_epoch = 0

def load_dataset():
    with open("words-idfs.txt") as f:
        vocab_size = len(f.read().splitlines())
    train_data_reader = DataReader(
        data_path="data-train-tf-idf.txt",
        batch_size=50,
        vocab_size=vocab_size
    )
    test_data_reader = DataReader(
        data_path="data-test-tf-idf.txt",
        batch_size=50,
        vocab_size=vocab_size
    )
    return train_data_reader, test_data_reader, vocab_size

def save_parameters(name, value, epoch):
    filename = name.replace(":", "-colon-") + "-epoch-{}.txt".format(epoch)
    if len(value.shape) == 1: # is a list
        string_form = ','.join([str(number) for number in value])
    else: # is a list of lists
        string_form = '\n'.join([','.join([str(number) for number in value[row]]) for row in range(value.shape[0])])
    with open("saved-paras/" + filename, "w") as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    with open('saved-paras/' + filename, "r") as f:
        lines = f.read().splitlines()
    if len(lines) == 1: # is a vector
        value = [float(number) for number in lines[0].split(',')]
    else: # is a matrix
        value = [[float(number) for number in lines[row].split(',')] for row in range(len(lines))]
    return np.array(value)

def main():
    # create a computation graph
    train_data_reader, test_data_reader, vocab_size = load_dataset()
    step, MAX_STEP = 0, 1000

    model = MLP(vocab_size=vocab_size, hidden_size=50, num_classes=20)
    model.build_graph()

    # training
    while step < MAX_STEP:

        train_data, train_labels = train_data_reader.next_batch()
        loss = model.train_step(X=train_data,
                                y=train_labels,
                                learning_rate=0.01)
        step += 1
        if step % 100 == 0:
            print(f"Step: {step} - Loss: {loss.numpy()}")

    # save parameters
    trainable_variables = model._trainable_variables
    for variable in trainable_variables:
        save_parameters(name=variable.name,
                        value=variable.numpy(),
                        epoch=train_data_reader._num_epoch)
    
    # reset parameters
    model.build_graph()

    # restore parameters
    for variable in model._trainable_variables:
        saved_value = restore_parameters(variable.name, epoch=train_data_reader._num_epoch)
        if saved_value.ndim == 1:
            variable.assign(saved_value.reshape(1,saved_value.shape[0]))
        else:
            variable.assign(saved_value)
    
    # analyze model on train data
    train_data_reader.reset()
    num_true_preds_train = 0
    while True:
        train_data, train_labels = train_data_reader.next_batch()
        train_plabels_eval = model.predict(train_data)
        train_matches = np.equal(train_plabels_eval, train_labels)
        num_true_preds_train += np.sum(train_matches.astype(float))

        if train_data_reader._batch_id == 0:
            break
    print('Accuracy on train data:', num_true_preds_train / test_data_reader._data.shape[0])
    
    # analyze model on test data
    num_true_preds_test = 0
    while True:
        test_data, test_labels = test_data_reader.next_batch()
        test_plabels_eval = model.predict(test_data)
        test_matches = np.equal(test_plabels_eval, test_labels)
        num_true_preds_test += np.sum(test_matches.astype(float))

        if train_data_reader._batch_id == 0:
            break
    print('Accuracy on test data:', num_true_preds_test / test_data_reader._data.shape[0])

if __name__ == '__main__':
    main()