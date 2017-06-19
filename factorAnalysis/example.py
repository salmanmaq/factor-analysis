# This is an example script for Factor Analysis

# Import Libraries
import numpy as np
from fa import factor_analysis
import matplotlib.pyplot as plt

# Sample random data
# Each column is one example. For example
# X = np.random.randn(2000, 1000) will
# produce 1000 samples of each of which are
# 2000-dimensional
X = np.random.randn(2000, 1000)

# Calculate the mean and the Standard distribution
# of the original distribution
mu = np.mean(X)
sigma = np.std(X)
print('\nMean of original distribution= %f' %mu)
print('Standard Deviation of original distribution= %f\n' %sigma)

# Specify some hyper-parameters
k = 100 # The dimensionality of the projected space
itr = 200 # The number of EM iterations needed

# Do the factor analysis
z = factor_analysis(X, k, itr)

# Calculate the mean and the variance of the projection
mu_z = np.mean(z)
sigma_z = np.std(z)
print('Mean of the projection = %f' %mu_z)
print('Standard Deviation of the projection = %f' %sigma_z)

# Plot the original and projected dataset distribution (Just the first 2 examples)
xmin = int(1.5 * np.min(X))
xmax = int(1.5 * np.max(X))

plt.figure(1)
plt.subplot(121, aspect='equal')
plt.scatter(X[:,0],X[:,1])
orig_title = 'The Original Distribution.\n Mean = %f, Std = %f' %(mu, sigma)
plt.title(orig_title)
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)

plt.subplot(122, aspect='equal')
plt.scatter(z[:,0],z[:,1], c='r')
proj_title = 'The Projected Distribution.\n Mean = %f, Std = %f' %(mu_z, sigma_z)
plt.title(proj_title)
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)

plt.tight_layout()
plt.show()
