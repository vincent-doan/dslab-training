import numpy as np

def read_file(file):
    X = list()
    Y = list()

    with open(file) as fp:
        for line in fp:
            line = line.split()
            X.append(list(map(float,line[1:-1])))
            Y.append(float(line[-1]))

    X = np.array(X) #60 x 15 matrix
    Y = np.array(Y) #60 x 1 matrix

    return (X,Y)

def feature_scaling_and_add_one(X):
    X_max = np.array([np.amax(X[:, column_id]) for column_id in range(X.shape[1])])
    X_min = np.array([np.amin(X[:, column_id]) for column_id in range(X.shape[1])])

    X_scaled = (X - X_min) /(X_max - X_min)

    ones = np.array([[1] for _ in range(X_scaled.shape[0])])

    return np.column_stack((ones, X_scaled))

class RidgeRegression:
    def __init__(self):
        return
    
    def compute_RSS(self, Y_predicted, Y_new):
        loss = 1/Y_new.shape[0] * np.sum((Y_predicted - Y_new) ** 2)
        return loss
    
    def predict(self, W, X_new):
        return X_new.dot(W)
    
    def fit_ne(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]

        # normal equation formula to compute the optimizing weights, inefficient with large datasets
        W = np.linalg.inv(X_train.transpose().dot(X_train) + LAMBDA * np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W
    
    def fit_gd(self, X_train, Y_train, LAMBDA, learning_rate, max_num_epoch=100, batch_size=128):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        
        # initiate weights for each attribute, this one returns a sample from the standard normal dist.
        W = np.random.randn(X_train.shape[1])
        # set initial loss to positive infinity, this one is a arbitrarily large number
        prev_loss = 10e+8
        # set threshold to stop going into next epoch
        e = 1e-5

        for ep in range(max_num_epoch):
            # for each epoch, we shuffle the dataset we have, then we parse through them all, without repeats, in one epoch
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]

            total_num_minibatch = int(np.ceil(X_train.shape[0] / batch_size))

            for i in range(total_num_minibatch):
                # picking out a minibatch from top to bottom of shuffled dataset
                index = i * batch_size
                X_train_sub = X_train[index : index + batch_size]
                Y_train_sub = X_train[index : index + batch_size]
                grad = X_train_sub.T.dot(X_train_sub.dot(W) - Y_train_sub) + LAMBDA * W
                W = W - learning_rate * grad
            
            # update loss at the end of each epoch
            new_loss = self.compute_RSS(self.predict(W,X_train), Y_train)
            if (np.abs(new_loss - prev_loss)) <= e:
                break

            last_loss = new_loss
        
        return W            

    def get_best_LAMBDA(self, X_train, Y_train):

        def cross_validation(num_folds, LAMBDA):
            row_ids = np.array(range(X_train.shape[0]))
            # remove the some rows at the bottom of X_train to maintain minibatches of equal length (np.split() requires so)
            # valid_ids is a numpy array containing folds
            valid_ids = np.split(row_ids[: len(row_ids) - len(row_ids) % num_folds], num_folds)
            # add the remaining bottom rows to the final fold
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds :])
            # train_ids is a numpy array containing 'the rest of data' respective to each fold
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            
            aver_RSS = 0
            for i in range(num_folds):
                # for each iteration, pick one fold to be the cross-validating set, and the remaining data becomes the training set
                X_valid_fcv = X_train[valid_ids[i]]
                Y_valid_fcv = Y_train[valid_ids[i]]
                X_train_fcv = X_train[train_ids[i]]
                Y_train_fcv = Y_train[train_ids[i]]
                
                # we use fit_ne (normal equation) here to avoid having to consider about the learning rate
                W = self.fit_ne(X_train_fcv, Y_train_fcv, LAMBDA)
                aver_RSS += self.compute_RSS(self.predict(W, X_valid_fcv), Y_valid_fcv)
            
            return aver_RSS / num_folds
        
        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            # given a list of LAMBDA values to test, we try them out and select one with the lowest average loss
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return (best_LAMBDA, minimum_RSS)
        
        # obtain the integer part of the LAMBDA value first, then try to specify it further
        (best_LAMBDA, minimum_RSS) = range_scan(best_LAMBDA=0, minimum_RSS=1e+10, LAMBDA_values=list(range(50)))
        
        # specifying the decimal part
        step_size = 0.001
        LAMBDA_values = [(best_LAMBDA-1) + step_size * k for k in range(int(2/step_size))]
        (best_LAMBDA, minimum_RSS) = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS, LAMBDA_values=LAMBDA_values)

        return best_LAMBDA

def main():
    (X,Y) = read_file("death-rate.txt")
    
    X = feature_scaling_and_add_one(X)
    # divide training set and test set
    X_train = X[:50]
    Y_train = Y[:50]
    X_test = X[50:]
    Y_test = Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_best_LAMBDA(X_train, Y_train)
    print('Best LAMBDA:',best_LAMBDA)
    W_learned = ridge_regression.fit_ne(X_train, Y_train, best_LAMBDA)
    print('Loss:',ridge_regression.compute_RSS(ridge_regression.predict(W_learned, X_test), Y_test))

if __name__ == "__main__":
    main()