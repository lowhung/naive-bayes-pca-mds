import numpy as np
from numpy.linalg import norm
import utils
from pca import PCA
from utils import find_min
from scipy import stats

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        Z = PCA(k).fit(X).compress(X)

        # Solve for the minimizer
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()

class ISOMAP(MDS):

    def __init__(self, n_components, n_neighbours):
        self.k = n_components
        self.nn = n_neighbours

    def knn(self, X):
        squared_distances = utils.euclidean_dist_squared(X,X)
        squared_distances = np.sqrt(squared_distances)
        rows, cols = squared_distances.shape
        nearest = np.zeros((X.shape[0], self.nn+1), dtype=int)

        for i in range(rows):
            # sort the distances to other points
            inds = np.argsort(squared_distances[:,i])
            # compute mode of k closest training pts
            nearest[i] = inds[:self.nn+1]
        return nearest

    def compress(self, X):
        n = X.shape[0]
        # nearest_neighbours = np.zeros((n, self.nn))

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.

        adjacency_matrix = np.zeros((n,n))
        nearest_neighbours = self.knn(X)
        for i, j in enumerate(nearest_neighbours):
            for neighbour in j:
                adjacency_matrix[i,neighbour] = D[i,neighbour]
                adjacency_matrix[neighbour,i] = D[neighbour,i]

        dijkstra = utils.dijkstra(adjacency_matrix)

        dijkstra[np.isinf(dijkstra)] = dijkstra[~np.isinf(dijkstra)].max()
        # Initialize low-dimensional representation with PCA
        Z = PCA(self.k).fit(X).compress(X)

        # Solve for the minimizer
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, dijkstra)
        Z = z.reshape(n, self.k)
        return Z
