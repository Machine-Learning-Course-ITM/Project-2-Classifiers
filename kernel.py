import numpy as np

### Functions for you to fill in ###


def linear_kernel(X, Y):
    """
        Compute the linear kernel between two matrices X and Y::
            K(x, y) = <x, y>
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return np.dot(X, Y.T)

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    K = np.empty((X.shape[0], Y.shape[0]))
    np.dot(X, Y.T, out=K)
    np.add(K, c, out = K)
    np.power(K, p, out=K)
    return K



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    K = np.empty((X.shape[0], Y.shape[0]))
    # X2 = X.reshape((X.shape[0], 1, X.shape[1]), order='F')
    # Y2 = Y.reshape((1, Y.shape[0], Y.shape[1]), order='F')
    # temp = X2 - Y2
    # np.power(temp, 2, out=temp)
    # np.sum(temp, axis=2, out=K)
    # np.multiply(K, -gamma, out=K)
    # np.exp(K, out=K)
    # return K
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i,j] = np.exp(-gamma*((X[i,:] - Y[j,:])**2).sum()) 
    return K

