import numpy as np

class pptk:

    def min_max_scaler(self,data):
        # Normalize the data using min-max scaling
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
        return (data - min_values) / (max_values - min_values)
    
    def shuffle(self,data):
        # Data splitting
        np.random.seed(42)
        # Genearate indices for shuffle.
        indices = np.random.permutation(len(data))
        return indices
    
    def train_test_split(self,data,labels):
        # set the train, test split
        split = int(0.8 * len(data))
        # shuffle the data and set the train and test indices
        indices = self.shuffle(data)
        train_indices, test_indices = indices[:split], indices[split:]
        # apply the index split to train and test sets.
        X_train, X_test = data[train_indices], data[test_indices]
        # apply the index split to train and test labels.
        y_train, y_test = labels[train_indices], labels[test_indices]
        return X_train, X_test, y_train, y_test
    
    def one_hot_encode(self,no_labels,y_train,y_test):
        # One hot encodes training and test labels
        y_train_one_hot = np.eye(no_labels)[y_train]
        y_test_one_hot = np.eye(no_labels)[y_test]
        return y_train_one_hot,y_test_one_hot