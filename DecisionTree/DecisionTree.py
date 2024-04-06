# _*_ coding : utf-8 _*_
# @Time : 2024/04/06 10:31
# @Author : momo
# @File : DecisionTree
# @Project : MLBaseAlgorithm


import math

import numpy as np


class FieldName:
    feature = "feature"
    threshold = "threshold"
    left = "left"
    right = "right"
    class_ = "class"


class DecisionTree:
    def __init__(self):
        self.X = None
        self.y = None
        self.tree = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        y = []
        for sample in X:
            y.append(self._predict_single(sample, self.tree))
        return y

    def _grow_tree(self, X, y):
        # 剩一个类别，直接返回本节点
        if len(set(y)) == 1:
            return {FieldName.class_: y[0]}
        # 无特征，返回最大类别
        if len(X[0]) == 0:
            y, counts = np.unique(y, return_counts=True)
            return {FieldName.class_: y[np.argmax(counts)]}

        # 选取最佳特征和最佳阈值
        best_feature, best_threshold = self._find_best_split(X, y)

        # 根据最佳进行划分得到左右子树
        X_left, y_left, X_right, y_right = self._best_split(X, y, best_feature, best_threshold)

        left_tree = self._grow_tree(X_left, y_left)
        right_tree = self._grow_tree(X_right, y_right)

        return {
            FieldName.feature: best_feature,
            FieldName.threshold: best_threshold,
            FieldName.left: left_tree,
            FieldName.right: right_tree
        }

    def _find_best_split(self, X, y):
        best_threshold = None
        best_feature = None
        # 参考
        min_gini = math.inf

        X = np.array(X)
        y = np.array(y)
        for feature in range(X.shape[1]):
            for threshold in set(X[:, feature]):
                # 分边
                y_left = []
                y_right = []
                for i in range(X.shape[0]):
                    if X[i, feature] <= threshold:
                        y_left.append(y[i])
                    else:
                        y_right.append(y[i])
                # gini
                gini = self._calculate_gini(y_left, y_right)

                if gini < min_gini:
                    min_gini = gini
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold

    def _calculate_gini(self, y_left, y_right):
        # p
        n1 = len(y_left)
        n2 = len(y_right)
        n = n1 + n2

        # 分别数量
        _, left_counts = np.unique(y_left, return_counts=True)
        _, right_counts = np.unique(y_right, return_counts=True)

        # 分别gini
        gini_left = 1 - sum((left_counts / n1) ** 2)
        gini_right = 1 - sum((right_counts / n2) ** 2)

        # 加权
        gini = (n1 / n) * gini_left + (n2 / n) * gini_right

        return gini

    def _best_split(self, X, y, best_feature, best_threshold):
        X_left = []
        y_left = []
        X_right = []
        y_right = []

        for i in range(len(X)):
            if X[i][best_feature] <= best_threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        return X_left, y_left, X_right, y_right

    def _predict_single(self, X, tree):
        # 类别节点
        if FieldName.class_ in tree:
            return tree[FieldName.class_]

        feature = tree[FieldName.feature]
        threshold = tree[FieldName.threshold]

        if X[feature] <= threshold:
            return self._predict_single(X, tree[FieldName.left])
        else:
            return self._predict_single(X, tree[FieldName.right])


if __name__ == '__main__':
    # 创建一个样本数据集
    X_train = [[6.7, 3.1, 4.7, 1.5],
               [5.1, 3.5, 1.4, 0.2],
               [5.7, 2.8, 4.1, 1.3],
               [6.3, 2.9, 5.6, 1.8],
               [4.9, 3.1, 1.5, 0.1],
               [6.4, 3.2, 4.5, 1.5],
               [5.8, 2.7, 5.1, 1.9],
               [5.1, 3.8, 1.5, 0.3],
               [4.7, 3.2, 1.6, 0.2],
               [6.3, 3.3, 6.0, 2.5]]

    y_train = [1, 0, 1, 2, 0, 1, 2, 0, 0, 2]  # 样本对应的类别

    # 创建决策树模型并训练
    model = DecisionTree()
    model.fit(X_train, y_train)

    # 创建一个测试数据集
    X_test = [[5.6, 2.8, 4.9, 2.0],
              [4.8, 3.0, 1.4, 0.3],
              [6.1, 3.0, 4.6, 1.4]]

    # 使用训练好的模型进行预测
    predictions = model.predict(X_test)

    # 输出预测结果
    print("预测结果:", predictions)
