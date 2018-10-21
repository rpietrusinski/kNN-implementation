import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class KnnClassifier(object):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    def standardize(self, vector):
        mean = np.mean(vector)
        sd = np.std(vector)
        return np.array([(x - mean) / sd for x in vector])

    def prepare_data(self):
        X_scaled = np.apply_along_axis(self.standardize, 0, self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, self.y, test_size=0.33,
                                                                                random_state=42)

    def majority_vote(self, lst):
        return max(set(lst), key=lst.count)

    def single_pred(self, vec, neighbors):
        most_similar_indices = [i[0] for i in sorted(enumerate(vec), key=lambda x: x[1])[:neighbors]]
        prediction = self.majority_vote(list(self.y_train[most_similar_indices]))
        return prediction

    def predict(self, test_data, neighbors):
        obs_train = self.X_train.shape[0]
        obs_test = test_data.shape[0]

        # distance matrix
        distance_matrix = np.zeros((obs_train, obs_test))
        distance_matrix[:] = np.nan
        for i in range(obs_train):
            for j in range(obs_test):
                vec_i = self.X_train[i, :]
                vec_j = test_data[j, :]
                euc = np.sqrt(np.sum([(p - q) ** 2 for p, q in zip(vec_i, vec_j)]))
                distance_matrix[i, j] = euc
        return np.apply_along_axis(self.single_pred, 0, distance_matrix, neighbors)


# load data
data = load_breast_cancer()
X = data['data']
y = data['target']


# Experiment with different values for number of neighbors
def run_experiment(num_neighbors):
    knn = KnnClassifier(X, y)
    knn.prepare_data()
    train_preds = knn.predict(knn.X_train, num_neighbors)
    test_preds = knn.predict(knn.X_test, num_neighbors)

    train_accuracy = np.mean(train_preds == knn.y_train)
    test_accuracy = np.mean(test_preds == knn.y_test)
    print("Number of neighbors: {}".format(num_neighbors))
    print("-----------------------")
    print("Train accuracy = {:.2%}".format(train_accuracy))
    print("Test accuracy = {:.2%}".format(test_accuracy))
    print("\n")
    return train_accuracy, test_accuracy


values = np.arange(2, 21)
resu = list(map(run_experiment, values))
train = [x[0] for x in resu]
test = [x[1] for x in resu]

# Plot kNN vs accuracy
plt.figure(figsize=(12, 8))
plt.plot(values, train, label="Train")
plt.plot(values, test, label="Test")
plt.legend()
plt.title('Train vs test accuracy')
plt.xlabel("k-nearest neighbors")
plt.ylabel('Accuracy')
plt.savefig('plot1.jpg')

# PCA
knn = KnnClassifier(X, y)
knn.prepare_data()
preds = knn.predict(knn.X_train, 9)
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(knn.X_train)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title("True labels")
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=knn.y_train)
plt.subplot(1, 2, 2)
plt.title("Predicted labels")
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=preds)
plt.suptitle("PCA - First 2 comp. Overall accuracy = 97.4%")
plt.savefig('plot2.jpg')
