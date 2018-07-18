import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Datasets imported
#dataset_train for training
#dataset_test for testing
dataset_train = pd.read_csv(path_train)
dataset_test = pd.read_csv(path_test)

#X consists of 3 features
#Y consists of a single output feature

#creates X dataframe for training
dataset_train_x = dataset_train.drop(['y'], axis=1)

#creates Y dataframe for training
dataset_train_y = dataset_train[['y']]

dataset_train.head() #shows first lines of data in the dataset
dataset_test.head() 

#asserting additional information on training and test sets
dataset_train.info()
dataset_test.info()

#evaluate general statistical parameters
dataset_train.describe()
dataset_test.describe()

#input plots
#x1 - same approach for other variables
array = np.array(dataset_train.x1.values)
n, histo, pat = plt.hist(array,100,normed=1)
mu = np.mean(array)
sigma = np.std(array)

plt.plot(histo, mlab.normpdf(histo,mu,sigma))
plt.title('Fig.1. Histogram of variable x1 - Train')
plt.show()

#plotting the probability distribution function of input variables allows the better understanding of the variable behavior

#input space
#3d plot of two chosen input variables and output
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(dataset_train.x1, dataset_train.x2, dataset_train.y, '.')
plt.title('Fig.2. Input variables space')
