
#Datasets imported

#X consists of 3 features
#Y consists of a single output feature

#Check train and test data for NaN
dataset_train.isnull().values.any()
dataset_test.isnull().values.any()

# Input-Output distinction
dataset_train_x = dataset_train.copy(deep=True)
dataset_train_x = dataset_train_x.drop(['y'], axis=1)
dataset_train_y = dataset_train[['y']]

#DATA ENGINEERING

dataset_train_x['mode'] = X_train.mode(axis=1)
dataset_train_x['sum'] = X_train.sum(axis=1, skipna=True)
dataset_train_x['mean'] = X_train.mean(axis=1, skipna=True)
dataset_train_x['std'] = X_train.std(axis=1, skipna=True)
dataset_train_x['z_std'] = X_train_nan.std(axis=1, skipna=True)
dataset_train_x['z_mean'] = X_train_nan.mean(axis=1, skipna=True)
dataset_train_x['median'] = X_train_nan.median(axis=1, skipna=True)
dataset_train_x['max'] = X_train.max(axis=1, skipna=True)
dataset_train_x['min'] = X_train_nan.min(axis=1, skipna=True)
dataset_train_x['var'] = X_train_nan.var(axis=1)

dataset_train_x['x1x2'] = np.multiply(dataset_train_x.x1,dataset_train_x.x2)
dataset_train_x['x1x2^2'] = dataset_train_x['x1x2']**2
dataset_train_x['x1x2^3'] = dataset_train_x['x1x2']**3
dataset_train_x['x1/x2'] = np.divide(dataset_train_x.x1,dataset_train_x.x2)
dataset_train_x['x1x2x3'] = np.multiply(dataset_train_x['x1x2'],dataset_train_x['x3'])
dataset_train_x['invx2'] = (dataset_train_x['x2']**(-1))
dataset_train_x['loginvx2'] = np.log(abs(dataset_train_x['invx2']))

# creating sample new variable from observating x1x2 interaction and PDF
dataset_train_x['newx'] = np.array((dataset_train.x2*2000)-dataset_train.x1.values)
dataset_train_x['newx2'] = (dataset_train_x['newx']**2)
dataset_train_x['newx3'] = (dataset_train_x['newx']**3)
dataset_train_x['newx4'] = (dataset_train_x['newx']**4)
dataset_train_x['lognewx'] = np.log(abs(dataset_train_x['newx']))
dataset_train_x['expnewx'] = np.exp(abs(dataset_train_x['newx']))
