import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100, gif_folder_path=None):
        self.weights = np.ones(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []
        self.accuracies = []
        self.gif_folder_path = gif_folder_path


    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        plt.ion()
        fig, ((ax1, ax4), (ax3, ax2)) = plt.subplots(2, 2, figsize=(8, 8))

        
        for epoch in range(self.epochs):
            total_loss = 0
            correct_predictions = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_loss += error**2
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
                if prediction == label:
                    correct_predictions += 1
            
            self.losses.append(total_loss / len(labels))
            self.accuracies.append(correct_predictions / len(labels))
            
            self.plot_decision_boundary(training_inputs, labels, ax1)
            ax1.set_title(f'Decision Boundary (Epoch: {epoch + 1})')
            
            
            ax2.clear()
            ax2.plot(range(1, epoch + 2), self.losses, 'r-')
            ax2.set_title('Loss Curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            
            ax3.clear()
            ax3.plot(range(1, epoch + 2), self.accuracies, 'b-')
            ax3.set_title('Accuracy Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')

            ax4.clear()
            x = range(len(self.weights))
            ax4.bar(x, self.weights)
            ax4.set_title('Weight Values')
            ax4.set_xlabel('Weight Index')
            ax4.set_ylabel('Weight Value')
            ax4.set_xticks(x)
            ax4.set_xticklabels(['Bias'] + [f'W{i}' for i in range(1, len(self.weights))])
            
            if self.gif_folder_path:
                plt.savefig(f'{self.gif_folder_path}/frame_{epoch:04d}.png')

            plt.tight_layout()
            plt.pause(0.1)
        
        plt.ioff()
        plt.show()

    def plot_decision_boundary(self, training_inputs, labels, ax):
        ax.clear()
        x_min, x_max = training_inputs[:, 0].min() - 1, training_inputs[:, 0].max() + 1
        y_min, y_max = training_inputs[:, 1].min() - 1, training_inputs[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = np.array([self.predict(np.array([xi, yi])) for xi, yi in zip(np.ravel(xx), np.ravel(yy))])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax.scatter(training_inputs[:, 0], training_inputs[:, 1], c=labels, cmap=plt.cm.RdYlBu, edgecolor='black')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

# Example usage:
# if __name__ == "__main__":
#     # Define a simple dataset (XOR gate)
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([0, 0, 0, 1])

#     # Create and train the perceptron
#     perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100, gif_folder_path='folder_path_here')
#     perceptron.train(X, y)
