import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

#reads both datasets
dataset_train = pd.read_csv(path_train)
dataset_test = pd.read_csv(path_test)

# initializes different dictionaries for integer and categorical data
dcat = defaultdict(LabelEncoder)
dint = defaultdict(LabelEncoder)

# remember to change categorical data mix with nums

# keep an eye for discrepancies on dataframes dtypes

cat_columns = x_test.select_dtypes(['object']).columns.tolist() #gets all the columns that are objects
int_columns = x_test.select_dtypes(['int64']).columns.tolist() #gets all the columns that are integers


# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
# Encoding the variable 
x_cat=pd.concat((x_train,x_test),axis=0) #creates a big vector for all training X data
x_cat[cat_columns].apply(lambda x: dcat[x.name].fit(x))
x_cat[int_columns].apply(lambda x: dint[x.name].fit(x))

# Using the dictionary to label future data
x_train[cat_columns] = x_train[cat_columns].apply(lambda x: dcat[x.name].transform(x))
x_train[int_columns] = x_train[int_columns].apply(lambda x: dint[x.name].transform(x))

x_test[cat_columns] = x_test[cat_columns].apply(lambda x: dcat[x.name].transform(x))
x_test[int_columns] = x_test[int_columns].apply(lambda x: dint[x.name].transform(x))

# Inverse the encoded variables given that the column names are the same
#x_train.apply(lambda x: d[x.name].inverse_transform(x))





