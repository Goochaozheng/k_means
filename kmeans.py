import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default="iris")
parser.add_argument("-k", type=int, default=2)
parser.add_argument("-max_it", type=int, default=20)


class kmeans:
    def __init__(self, max_it=100, k=3):
        print("K: %d" % k)
        print("Max Iter: %d" % max_it)
        self.num_cluster = k
        self.max_iter = max_it

    # main iteration of solving k-means
    def fit(self, data):
        # get initial label
        data_dim = data.shape[1]
        self.centroids = np.random.rand(self.num_cluster, data_dim)
        self.labels = np.empty(len(data))
        for i in range(len(data)):
            self.labels[i] = self.get_label(data[i])

        count = 0
        for e in range(self.max_iter):
            print("# Iter %d:" % e)
            print("# Centroids:")
            print(self.centroids)
            print("\n")

            # compute new centroid
            prev_centroids = self.centroids.copy()
            for k in range(self.num_cluster):
                clu = data[self.labels == k]
                self.centroids[k] = self.find_centroid(clu)

            # if the centroid stable, stop
            if (self.centroids == prev_centroids).all():
                break

            # assign new label according to current centroid
            for i in range(data.shape[0]):
                self.labels[i] = self.get_label(data[i])

            count += 1

        if self.max_iter != count:
            print("###########################################")
            print("Centroids stable, stop.")
        else:
            print("###########################################")
            print("Max iteration reached, stop.")

        print("Result centroids: ")
        print(self.centroids)
        print("Variance: %.4f" % np.mean(self.get_variance(data)))

        return

    # compute distance to centroids
    def distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    # calculate the centroid of given data points
    def find_centroid(self, x):
        return np.sum(x, axis=0)/len(x)

    # find the nearest centroid of a data point
    def get_label(self, x):
        res = 0
        min_dis = float("inf")
        for k in range(self.num_cluster):
            dis = self.distance(x, self.centroids[k])
            if dis < min_dis: 
                res = k
                min_dis = dis
        return res

    # calculate the average distance to the centroid for each cluster
    def get_variance(self, data):
        res = []
        for k in range(self.num_cluster):
            cluster_sample = data[self.labels == k]
            dis = 0
            for i in range(len(cluster_sample)):
                dis += self.distance(cluster_sample[i], self.centroids[k])
            res.append(dis/len(cluster_sample)) 

        return res


def main():
    args = parser.parse_args()

    # Load data
    print("Loading dataset: ", args.dataset)
    if args.dataset == "breast_cancer":
        breast = np.genfromtxt("data/breast-cancer-wisconsin.data", delimiter=",")
        # Remove nan data samples
        breast = breast[~np.isnan(breast).any(axis=1)][:,1:]
        np.random.shuffle(breast)
        data = breast[:,:-1]

    elif args.dataset == "iris":
        iris_data = np.loadtxt("data/iris.csv", delimiter=',', usecols=[0,1,2,3])
        data = iris_data

    else:
        raise Exception("Dataset can only be breast_cancer or energy_efficiency")

    # Standardize
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - min(data[:, i])) / (max(data[:, i]) - min(data[:, i]))

    # Create model and fit
    model = kmeans(k=args.k, max_it=args.max_it)
    model.fit(data)

if __name__ == "__main__":
    main()