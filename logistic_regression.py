import numpy             as np
import matplotlib.pyplot as plt

from sklearn.datasets  import make_blobs as blobs
from matplotlib.colors import ListedColormap


class LogisticRegerssion:

    colormap = ListedColormap(colors = ['#ff9999', '#19ff19'])
    colors   = ("red", "green")
    
    def __init__(self, max_iter : int, learning_rate : float) -> None:
        self.max_iterations = max_iter
        self.learning_rate  = learning_rate


    def logit(self, coords, weight) -> np.ndarray:
        return np.dot(coords, weight)
    

    def predict(self, coords : np.ndarray, threshold : float) -> np.ndarray:
        return self.proba(coords) >= threshold


    def proba(self, coords : np.ndarray) -> np.ndarray:
        n, _ = coords.shape
        enhanced_coords = np.concatenate((np.ones((n, 1)), coords), axis=1)

        return LogisticRegerssion.sigmoid(self.logit(enhanced_coords, self.weight))


    def run(self, coordinates : np.ndarray, samples : np.ndarray) -> None:
        self.samples_amount    = samples.size
        self.graph_coordinates = coordinates

        n, k             = coordinates.shape
        self.coordinates = np.concatenate((np.ones((n, 1)), coordinates), axis=1)
        self.weight      = np.random.normal(loc=0.0, scale=0.01, size=k + 1)

        for epoch in range(self.max_iterations):
            order = np.random.permutation(len(self.coordinates))

            for start_index in range(0, len(self.coordinates), self.samples_amount):

                batch_indices = order[start_index:start_index + self.samples_amount]

                coords_batch = self.coordinates[batch_indices]
                labels_batch = samples         [batch_indices]

                prediction = LogisticRegerssion.sigmoid(self.logit(coords_batch, self.weight))
                loss       = LogisticRegerssion.BCE(prediction, labels_batch)
                gradient   = LogisticRegerssion.gradient_sigmoid(coords_batch, prediction, labels_batch)

                self.weight -= gradient * self.learning_rate

                if epoch % 100 == 0: print(f"Итерация № {str(epoch)}; LOSS: {str(float(loss))}")
    

    def show(self, side_spacing : int, prediction_treshold : float) -> None:
        colored_labels = np.zeros(self.samples_amount, dtype=str)

        for i, cl in enumerate([0, 1]): colored_labels[labels == cl] = self.colors[i]

        x_coords, y_coords = np.meshgrid(
            np.linspace(np.min(self.graph_coordinates[:, 0]) - side_spacing, np.max(self.graph_coordinates[:, 0]) + side_spacing, 500), 
            np.linspace(np.min(self.graph_coordinates[:, 1]) - side_spacing, np.max(self.graph_coordinates[:, 1]) + side_spacing, 500)
        )

        predictional_division = LogReg.predict(np.c_[x_coords.ravel(), y_coords.ravel()], threshold=prediction_treshold).reshape(x_coords.shape)

        plt.figure(figsize=(9, 6))
        plt.title("Logistic regression")

        plt.pcolormesh(x_coords, y_coords, predictional_division, cmap=self.colormap)
        plt.scatter(self.graph_coordinates[:, 0], self.graph_coordinates[:, 1], c=colored_labels)
        plt.grid(alpha=0.5)
        plt.show()


    @staticmethod
    def sigmoid(logits : np.ndarray) -> np.ndarray:
        return 1. / (1 + np.exp(-logits))


    @staticmethod
    def BCE(pred : np.ndarray, labels : np.ndarray) -> np.ndarray:
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return np.mean(-(labels * np.log(pred) + (1 - labels) * np.log(1 - pred)))


    @staticmethod
    def gradient_sigmoid(coords : np.ndarray, pred : np.ndarray, labels : np.ndarray) -> np.ndarray:
        return np.dot(coords.T, (pred - labels)) / labels.size


coordinates, labels = blobs(n_samples=300, centers=[[2, 3], [-3, -1]], cluster_std=2, random_state=1520)

LogReg = LogisticRegerssion(max_iter=10000, learning_rate=0.001)
LogReg.run(coordinates, labels)
LogReg.show(side_spacing=2, prediction_treshold=0.5)