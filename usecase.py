from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from knn import KnnClassifier
import numpy as np


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