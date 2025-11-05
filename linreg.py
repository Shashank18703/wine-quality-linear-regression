import numpy as np

def load_data(path, num_train):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    X = data[:, :-1]
    Y = data[:, -1]
    X_train = X[:num_train,:]
    Y_train = Y[:num_train]
    X_test = X[num_train:,:]
    Y_test = Y[num_train:]
    
    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    A = X.shape[0]
    X_T = np.column_stack([X, np.ones(A)])
    theta = np.linalg.solve(X_T.T @ X_T, X_T.T @ Y)

    return theta


def predict(X, theta):
    B = X.shape[0]
    X_T = np.column_stack([X, np.ones(B)])
    Y_pred = X_T @ theta

    return Y_pred


def energy(Y_pred, Y_gt):
    se = np.sum((Y_pred - Y_gt)**2)

    return se
