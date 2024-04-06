import random
import math


# 先看懂python代码 再写np的
class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X_train):
        # 随机初始化质心
        self.centroids = random.sample(X_train, self.k)

        for _ in range(self.max_iters):
            # 分配样本到最近的质心
            clusters = [[] for _ in range(self.k)]
            for sample in X_train:
                i = self._nearest_centroid(sample)
                clusters[i].append(sample)

            # 更新质心
            new_centroids = [self._compute_mean(cluster) for cluster in clusters]

            # 如果质心不再改变，结束迭代
            if self.centroids == new_centroids:
                break

            self.centroids = new_centroids

    def predict(self, X_predicts):
        y_predicts = []
        for sample in X_predicts:
            i = self._nearest_centroid(sample)
            y_predicts.append(i)
        return y_predicts

    def _nearest_centroid(self, sample):
        min_distance = math.inf
        i = None
        for j, centroid in enumerate(self.centroids):
            distance = self._euclidean_distance(sample, centroid)
            if distance < min_distance:
                min_distance = distance
                i = j
        return i

    def _euclidean_distance(self, p1, p2):
        return math.sqrt(sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]))

    def _compute_mean(self, cluster):
        # if len(cluster) == 0:
        #     return random.choice(cluster)
        # 一列计算
        return [sum(coord) / len(coord) for coord in zip(*cluster)]


# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    # random.seed(0)
    X = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(100)]

    # 创建 KMeans 实例并指定簇的数量
    kmeans = KMeans(k=3)

    # 调用 fit 方法进行聚类
    kmeans.fit(X)

    # 打印结果
    print("质心坐标：", kmeans.centroids)

    # 预测新数据点的簇标签
    new_data = [(0.1, 0.2), (0.5, 0.6), (0.8, 0.9)]
    labels = kmeans.predict(new_data)
    print("新数据点的簇标签：", labels)