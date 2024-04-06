# _*_ coding : utf-8 _*_
# @Time : 2024/04/04 0:18
# @Author : momo
# @File : KNN
# @Project : MLBaseAlgorithm

import numpy as np


class KNNClassifier:
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_predicts):
        y_predicts = []
        for X_predict in X_predicts:
            distances = np.sqrt(np.sum(np.power(X_predict - self.X_train, 2), axis=1))
            neighbors_indices = np.argsort(distances)[:self.k_neighbors]
            neighbor_labels = self.y_train[neighbors_indices]
            unique_labels, label_counts = np.unique(neighbor_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(label_counts)]
            y_predicts.append(predicted_label)
        return y_predicts


if __name__ == '__main__':
    X_train = np.array(
        [
            [1, 2.5, 1.3, 2.4, 6.5],
            [1.1, 1.2, 2.1, 6, 2],
            [0.2, 0.1, 3.2, 4, 1],
            [1, 2.3, 0.3, 4.4, 5]
        ]
    )
    y_train = np.array([1, 0, 0, 1])

    knn = KNNClassifier(3)

    knn.fit(X_train, y_train)

    X_predict = np.array([[2, 3, 5, 3, 5]])
    y_predicted = knn.predict(X_predict)
    print(y_predicted)
