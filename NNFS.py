import numpy as np
import matplotlib.pyplot as plt

class NNFS:

    def __init__(self,input_size,hidden_in,hidden_out,output_size,epochs,batch_size,learning_rate):
        np.random.seed(42)

        #
        self.input_size = input_size
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out
        self.output_size = output_size

        #
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_1 = np.random.randn(self.input_size, self.hidden_in)
        self.bias_1 = np.zeros(self.hidden_in)
        self.weights_2 = np.random.randn(self.hidden_in, self.hidden_out)
        self.bias_2 = np.zeros(self.hidden_out)
        self.weights_3 = np.random.randn(self.hidden_out, self.output_size)
        self.bias_3 = np.zeros(self.output_size)

        #
        self.train_losses = []
        self.train_accuracies = []

    def relu(self,x):
        return np.maximum(0, x)

    def softmax(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def dense_layer(self,x, weights, bias, activation_function):
        return activation_function(np.dot(x, weights) + bias)

    def categorical_crossentropy(self,y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)

    def accuracy(self, y_true, y_pred):
        predicted_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.sum(predicted_labels == true_labels) / len(true_labels)
    
    def forward(self,X_batch):
        # Forward pass
        layer_1 = self.dense_layer(X_batch, self.weights_1, self.bias_1, self.relu)
        layer_2 = self.dense_layer(layer_1, self.weights_2, self.bias_2, self.relu)
        output_layer = self.dense_layer(layer_2, self.weights_3, self.bias_3, self.softmax)
        return layer_1, layer_2, output_layer
    
    def train(self,X_train,y_train):
        for epoch in range(self.epochs):
            for i in range(0, len(X_train), self.batch_size):
                # Mini-batch
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                layer_1, layer_2, output_layer = self.forward(X_batch)

                # Compute loss (cross-entropy)
                loss = self.categorical_crossentropy(y_batch, output_layer)

                # Compute batch accuracy
                acc = self.accuracy(y_batch, output_layer)

                # Backpropagation
                grad_output = (output_layer - y_batch) / len(X_batch)
                grad_layer_2 = np.dot(grad_output, self.weights_3.T)
                grad_layer_1 = np.dot(grad_layer_2, self.weights_2.T)

                # Update weights and biases
                self.weights_3 -= self.learning_rate * np.dot(layer_2.T, grad_output)
                self.bias_3 -= self.learning_rate * np.sum(grad_output, axis=0)

                self.weights_2 -= self.learning_rate * np.dot(layer_1.T, grad_layer_2 * (layer_2 > 0))
                self.bias_2 -= self.learning_rate * np.sum(grad_layer_2 * (layer_2 > 0), axis=0)

                self.weights_1 -= self.learning_rate * np.dot(X_batch.T, grad_layer_1 * (layer_1 > 0))
                self.bias_1 -= self.learning_rate * np.sum(grad_layer_1 * (layer_1 > 0), axis=0)
            # Keep track of training statistics
            self.train_losses.append(loss)
            self.train_accuracies.append(acc)
    
    def evaluate(self,X_test,y_test_one_hot):
        # Evaluate the model on the test set
        _, _, output_layer_test = self.forward(X_test)
        test_accuracy = self.accuracy(y_test_one_hot, output_layer_test)
        return test_accuracy
    
    def visualize(self):
        # Plotting training accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), self.train_accuracies, label='Training Accuracy', marker='o')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plotting training loss
        plt.subplot(1, 2, 2)
        plt.plot(range(self.epochs), self.train_losses, label='Training Loss', marker='o', color='r')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()