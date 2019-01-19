import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

class Classifier:
    def __init__(self, clf, name):
        self.name = name
        self.classifier = clf
        data = np.genfromtxt('chips.txt', delimiter=',')
        self.X = data[:, :2]
        self.Y = data[:, 2]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.33,
                                                                                random_state=42)

    def _search_params(self):
        return []

    def gen_params(self):
        params = self._search_params()
        print('Best params for ' + self.name + ': ', params)
        return params

    def print_plot(self):
        x_min, x_max = self.X[:, 0].min(), self.X[:, 0].max()
        y_min, y_max = self.X[:, 1].min(), self.X[:, 1].max()
        x, y = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))

        z = self.classifier.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
        plt.contourf(x, y, z, alpha=0.75, cmap=plt.cm.Accent)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, cmap=plt.cm.Accent)
        plt.title(self.name)
        plt.show()

    def get_metrics(self):
        pred_train = self.classifier.predict(self.X_train)
        pred_test = self.classifier.predict(self.X_test)
        return [precision_recall_fscore_support(y_true=self.Y_train, y_pred=pred_train, average='binary')[:2],
                precision_recall_fscore_support(y_true=self.Y_test, y_pred=pred_test, average='binary')[:2]]

    def run(self):
        params = self.gen_params()
        self.classifier.set_params(**params)
        self.classifier.fit(self.X_train, self.Y_train)
        self.print_plot()
        return self.get_metrics()


class SVM(Classifier):
    def search_best_params(self, Cs, gammas):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        search = GridSearchCV(self.classifier, param_grid={'kernel': kernels, 'C': Cs, 'gamma': gammas})
        search.fit(self.X_train, self.Y_train)
        return search.best_params_

    def _search_params(self):
        C = [10 ** p for p in range(-3, 2)]
        gamma = [10 ** p for p in range(-3, 2)]
        approximate = self.search_best_params(C, gamma)
        C = [approximate['C'] + E for E in np.linspace(-5, 5, 20)]
        gamma = [approximate['gamma'] + E for E in np.linspace(-1, 1, 20)]
        params = self.search_best_params(C, gamma)
        return params


class KNN(Classifier):
    def _search_params(self):
        n_neighbors_array = np.arange(1,10)
        search = GridSearchCV(self.classifier, param_grid={'n_neighbors': n_neighbors_array})
        search.fit(self.X_train, self.Y_train)
        return search.best_params_

_svm = SVM(svm.SVC(), 'SVM')
_knn = KNN( KNeighborsClassifier(), 'KNN')

svm_metrics = _svm.run()
knn_metrics = _knn.run()

table = pd.DataFrame([svm_metrics, knn_metrics], index=['Precision/recall train', 'Precision/recall test'], columns=['SVM','KNN'])
print(table)