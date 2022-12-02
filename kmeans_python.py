"""
K-means Python

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Lloyd's implementation

maybe early stop on inertia 

"""

import random
import math
from copy import deepcopy

from math import pow

from utils import generate_mat

random.seed(42)


def dissimilarity(p1, p2):
    """squared Euclidian distance"""
    return sum(pow(a - b, 2) for a, b in zip(p1, p2))


class KMeans:
    def __init__(self, k=2, niter=5, tol=0.0001):
        self.k = k
        self.tol = tol
        self.niter = niter

    def get_col_min_max(self, X):
        """Min and max of each feature"""
        n = len(X[0])
        mins, maxs = [math.inf] * n, [-math.inf] * n
        for row in X:
            for j in range(n):
                mins[j] = min(row[j], mins[j])
                maxs[j] = max(row[j], maxs[j])
        return mins, maxs

    def init_centroids(self, cmin, cmax):
        """Init with random val bet min and max of a feature column

        centroids: a list of size k of lists of size n
        """

        return [
            [random.uniform(cmin[i] + 1, cmax[i] - 1) for i in range(len(cmin))]
            for _ in range(self.k)
        ]

    def calculate_centroid(self, cluster, X):
        """for a cluster get the means per feature column"""
        means = [0] * len(X[0])
        for ind in cluster:
            row = X[ind]
            for j in range(len(row)):
                means[j] += row[j]
        # make safe for empty clusters
        return [x / len(cluster) if len(cluster) else 0 for x in means]

    def update_centroids(self, clusters, X):
        """get centroids for new clusters"""
        return [
            self.calculate_centroid(cluster, X) for i, cluster in enumerate(clusters)
        ]

    def get_diff_centroids(self, centroids, new_centroids):
        """frobenius norm of difference matrix"""
        s = sum(
            sum(pow(a - b, 2) for a, b in zip(old, new))
            for old, new in zip(centroids, new_centroids)
        )
        return math.sqrt(s)

    def get_clusters(self, X, centroids):
        """list of k lists of indeces"""
        clusters = [[] for _ in range(self.k)]

        for i in range(len(X)):
            min_dist, min_ind = math.inf, -1
            row = X[i]
            for cl_ind in range(self.k):
                centroid = centroids[cl_ind]
                dist = dissimilarity(centroid, row)
                if dist < min_dist:
                    min_dist, min_ind = dist, cl_ind
            clusters[min_ind].append(i)

        return clusters

    def fit(self, X):
        cmin, cmax = self.get_col_min_max(X)
        centroids = self.init_centroids(cmin, cmax)

        i = 0
        while i < self.niter:
            i += 1
            clusters = self.get_clusters(X, centroids)
            prev_centroids = deepcopy(centroids)
            # the k clusters contain row indeces
            centroids = self.update_centroids(clusters, X)
            diff = self.get_diff_centroids(centroids, prev_centroids)
            print(f"Inertia at iter {i} is {self.get_inertia(clusters, centroids, X)}")
            if diff < self.tol:
                break

        return clusters, centroids

    def predict(self, centroids, X_test):
        """
        centroids: List of k lists returned from a fit call

        ret: list of points assignments to cluster
        """
        predictions = []
        for i in range(len(X_test)):
            min_dist, min_ind = math.inf, -1
            row = X_test[i]
            for cl_ind in range(self.k):
                centroid = centroids[cl_ind]
                dist = dissimilarity(centroid, row)
                if dist < min_dist:
                    min_dist, min_ind = dist, cl_ind
            predictions.append(min_ind)

        return predictions

    def get_inertia(self, clusters, centroids, X):
        """the within cluster sum of squared errors over all clusters"""
        return sum(
            [
                sum(dissimilarity(X[i], centroid) for i in cluster)
                for cluster, centroid in zip(clusters, centroids)
            ]
        )


def main():

    X_train, X_test = generate_mat(10, 3), generate_mat(2, 3)

    kmeans = KMeans()
    # -----------------
    # standardize first - when used with euclidian distance
    # or get principal components - for dim red
    # test on data
    # know kmeans++
    # know minibatch
    clusters, centroids = kmeans.fit(X_train)

    print("predict on new test set", kmeans.predict(centroids, X_test))

    # ------------------------------Data and sklearn
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=2,
        cluster_std=0.5,
        shuffle=True,
        random_state=0,
    )
    print(y)
    # import matplotlib.pyplot as plt
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    # plt.close()
    from sklearn.cluster import KMeans as skKMeans

    km = skKMeans(
        n_clusters=3, init="random", n_init=1, max_iter=50, tol=1e-04, random_state=42
    )

    y_km = km.fit_predict(X)
    print(sum(y_km == y), sum(y_km != y))

    local_km = KMeans(k=3, niter=50)
    clusters, centroids = local_km.fit(X)
    # print(clusters)

    y_local = local_km.predict(centroids, X)
    print("local", sum(y_local == y), sum(y_local != y))

    # local implement acc is equal or better than sklearn

    # Centroids - same local and sklearn for k = 2
    print("local", centroids)
    print("sklearn", km.cluster_centers_)

    # iterative optimization: minimize the within cluster sum of squared errors - Lloyd's algo
    print("sklearn inertia: ", km.inertia_)
    print("local inertia:", local_km.get_inertia(clusters, centroids, X))  # correct

    # Code not safe for empty clusters


if __name__ == "__main__":
    main()
