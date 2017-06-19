# Import Libraries
import numpy as np
from diagonalize import diagonalize

def factor_analysis(X, k, itr):
    '''Function for perforaming Factor Analysis'''

    # Initialize the parameter matrices
    lmbda = np.random.randn(X.shape[0], k)
    psi = diagonalize(np.random.randn(X.shape[0], X.shape[0]))

    # Iterate and cyclically perform EM
    print('Starting the EM iterations\n\n')
    for i in range(itr):

        # Calculate the expectations
        beta = np.dot(lmbda.T, np.linalg.inv(psi + np.dot(lmbda, lmbda.T)))
        E_z_X = np.dot(beta, X)

        E_zz_X = np. eye(beta.shape[0], lmbda.shape[1]) - np.dot(beta, lmbda) + np.dot(E_z_X, E_z_X.T)

        # Maximize the expectations
        lmbda = np.dot(np.dot(X, E_z_X.T), np.linalg.inv(E_zz_X))
        psi = (1/X.shape[1]) * diagonalize(np.dot(X, X.T) - np.dot(lmbda, np.dot(E_z_X, X.T)))

        print('%d iterations completed!' %(i+1))

    print('EM Complete!\n\n')

    # Get the projections
    mean = np.zeros(psi.shape[0])
    U = np.random.multivariate_normal(mean, psi, X.shape[1])
    z = np.dot(np.linalg.pinv(lmbda), X - U.T)

    return z
