import numpy as np

class NaiveBayes():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        N, D = X.shape

        # Compute the number of class labels
        C = self.num_classes

        # Compute the probability of each class i.e p(y==c)
        y_counts = np.bincount(y)
        p_y = y_counts / N
        counts = np.sum(y_counts)

        # Compute the conditional probabilities i.e.
        # p(x(i,j)=1 | y(i)==c) as p_xy
        # p(x(i,j)=0 | y(i)==c) as p_xy
        p_xy = np.ones((D, C, 2))
        for c in range(C):
            for j in range(D):
                p_xy[j,c,0] = np.sum((X[y==c][:,j]==0))/y_counts[c]
                p_xy[j,c,1] = np.sum((X[y==c][:,j]==1))/y_counts[c]
        self.p_y = p_y
        self.p_xy = p_xy


    def predict(self, X):
        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy()
            for d in range(D):
                probs *= p_xy[d, :, X[n, d]]

            y_pred[n] = np.argmax(probs)

        return y_pred
