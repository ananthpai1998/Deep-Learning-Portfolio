import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=10000, use_bias=True, gif_folder_path=None):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.use_bias = use_bias
        self.gif_folder_path = gif_folder_path
        if use_bias:
            self.bias_hidden = np.random.rand(1, hidden_size)
            self.bias_output = np.random.rand(1, output_size)
        else:
            self.bias_hidden = np.zeros((1, hidden_size))
            self.bias_output = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.losses = []
        self.accuracies = []

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        if self.use_bias:
            self.hidden_input += self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        if self.use_bias:
            self.final_input += self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)
        
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        
        if self.use_bias:
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
        
    def train(self, X, y, animate=True):
        if animate:
            fig, ((ax1, ax4), (ax3, ax2)) = plt.subplots(2, 2, figsize=(10, 6))

            plt.ion()

            contour_plot = self.plot_decision_boundary(X, y, ax1)
            cbar = fig.colorbar(contour_plot, ax=ax1) 
            cbar.set_label('Output Value') 

            def update(frame):
                output = self.forward(X)
                loss = np.mean((y - output) ** 2)
                accuracy = np.mean(np.round(output) == y)
                
                self.losses.append(loss)
                self.accuracies.append(accuracy)

                self.backward(X, y, output)

                if frame % 100 == 0:
                    print(f'Epoch {frame}, Loss: {loss}, Accuracy: {accuracy}')

                ax1.clear()
                contour_plot = self.plot_decision_boundary(X, y, ax1)

                ax1.set_title(f'Decision Boundary (Epoch: {frame + 1})')

                ax2.clear()
                ax2.plot(self.losses, 'r-')
                ax2.set_title('Loss Curve')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')

                ax3.clear()
                ax3.plot(self.accuracies, 'b-')
                ax3.set_title('Accuracy Curve')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Accuracy')

                ax4.clear()
                weights = np.concatenate([self.weights_input_hidden.flatten(), self.weights_hidden_output.flatten()])
                x = range(len(weights))
                ax4.bar(x, weights)
                ax4.set_title('Weight Values')
                ax4.set_xlabel('Weight Index')
                ax4.set_ylabel('Weight Value')
                ax4.set_xticks([]) 

                plt.tight_layout()

                if self.gif_folder_path:
                    frame_path = os.path.join(self.gif_folder_path, f'frame_{frame:04d}.png')
                    plt.savefig(frame_path)


            anim = FuncAnimation(fig, update, frames=self.epochs, repeat=False, interval=1)
            plt.show(block=False)
            plt.pause(0.00001)

            
            input("Press [enter] to continue.")
        else:
            for epoch in range(self.epochs):
                output = self.forward(X)
                loss = np.mean((y - output) ** 2)
                accuracy = np.mean(np.round(output) == y)
                
                self.losses.append(loss)
                self.accuracies.append(accuracy)

                self.backward(X, y, output)

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}')
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            contour_plot = self.plot_decision_boundary(X, y, ax1)
            fig.colorbar(contour_plot, ax=ax1) 
            ax1.set_title('Final Decision Boundary')

            ax2.plot(self.losses, 'r-')
            ax2.set_title('Loss Curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')

            ax3.plot(self.accuracies, 'b-')
            ax3.set_title('Accuracy Curve')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')

            plt.tight_layout()
            plt.show()

    def plot_decision_boundary(self, X, y, ax):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        Z = np.array([self.forward(np.array([[x,y]])) for x,y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)

        levels = np.linspace(0, 1, 3)
        contour_plot = ax.contourf(xx ,yy ,Z ,levels=levels ,alpha=0.8 ,cmap=plt.cm.RdYlBu )
        ax.scatter(X[:,0],X[:,1],c=y.ravel(),cmap=plt.cm.RdYlBu ,edgecolor='black')

        return contour_plot


# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])


# mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=5000, use_bias=False, gif_folder_path='folder_path_here')
# mlp.train(X, y, animate=True)


# for inputs in X:
#     prediction = mlp.forward(inputs)
#     print(f"Input: {inputs}, Predicted Output: {prediction[0][0]:.4f}, Rounded: {np.round(prediction[0][0])}")
