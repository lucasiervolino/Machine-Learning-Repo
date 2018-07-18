import pandas as pd
import numpy as np

#loads datasets
X = pd.read_csv(path_X)

# X is composed of 16 numerical and categorical features and 40 categorical sequential answers to questions
# either A, B, C, D, E

#Label encoder dict
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
dcat = defaultdict(LabelEncoder)
dint = defaultdict(LabelEncoder)

# Encoding the variable 
x_cat=pd.concat((x_train_final,x_test_final),axis=0) #creates a big vector for all training X data
x_cat = x_cat.fillna('A')

x_cat[cat_columns_train].apply(lambda x: dcat[x.name].fit(x))
x_cat[int_columns].apply(lambda x: dint[x.name].fit(x))

# Using the dictionary to label future data
x_train_final[cat_columns_train] = x_train_final[cat_columns_train].apply(lambda x: dcat[x.name].transform(x))
x_train_final[int_columns] = x_train_final[int_columns].apply(lambda x: dint[x.name].transform(x))

y_train_final = y_train_final.apply(lambda x: dcat[x.name].transform(x))

x_test_final[cat_columns] = x_test_final[cat_columns].apply(lambda x: dcat[x.name].transform(x))
x_test_final[int_columns] = x_test_final[int_columns].apply(lambda x: dint[x.name].transform(x))

# To invert the encoded variables
#x_train.apply(lambda x: d[x.name].inverse_transform(x))

# approach described on https://stackoverflow.com/questions/47594861/predicting-a-multiple-time-step-forward-of-a-time-series-using-lstm
# concatenates X train and test
# creates y from shifted X
entireData = [x_train_final2, x_test_final2]
entireData = pd.concat(entireData)
entireData2 = entireData.values.reshape(entireData.values.shape[0], entireData.values.shape[1],1)
entireData3 = entireData2[16:,:,:] #selects only the sequential part of inputs. 16 mix of categorical and numerical inputs and 40 sequential inputs.

from hmmlearn import hmm

np.random.seed(9)

#initializing the model
modelmm = hmm.GaussianHMM(n_components=7, covariance_type="full")
# transition weights, assuming they are known
modelmm.startprob_ = np.array([0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19])
modelmm.transmat_ = np.array([[0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.11, 0.21, 0.21, 0.21, 0.21],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19],
                            [0.001, 0.049, 0.19, 0.19, 0.19, 0.19, 0.19]])
modelmm.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
modelmm.covars_ = np.tile(np.identity(2), (3, 1,1))
#predicted sample
X, Z = modelmm.sample(100)

#second approach
remodel = hmm.GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
X_markov = entireData3.reshape(entireData3.shape[0], entireData3.shape[1])
X_markov2 = pd.DataFrame(X_markov)
X_markov2 = X_markov2.values.flatten()
lengths =[]
#for j in range(np.size(X_markov,0))
for i in range(len(X_markov)):
    lengths.append(len(X_markov[i]))
#fit for every samples
remodel.fit(np.reshape(X_markov2,(len(X_markov2),1),lengths))
hmm.GaussianHMM(n_components=5).fit(X_markov2, lengths)

#fit for 1 sample
remodel.fit(np.reshape(X_markov[0],(np.size(X_markov,1),1),lengths))

X_submit_markov = np.reshape(X_submit,(len(X_submit),40))
X_submit_markov = pd.DataFrame(X_submit_markov)
X_submit_markov = X_submit_markov.values.flatten()

#q1 = np.reshape(X_submit_markov[0],(np.size(X_submit,1),1))
prediction = remodel.predict(np.reshape(X_submit_markov,(len(X_submit_markov),1),lengths[:len(X_submit)]))#lengths[:len(X_submit)]
pred_final = np.reshape(prediction,(len(X_submit),40))

#creates final vector
question_final = pd.DataFrame(pred_final[:,:5],columns=['Q41','Q42','Q43','Q44','Q45'])
question_final[question_final == 0] = 'A'
question_final[question_final == 1] = 'B'
question_final[question_final == 2] = 'C'
question_final[question_final == 3] = 'D'
question_final[question_final == 4] = 'E'
question_final = question_final.apply(lambda x: dcat[x.name].inverse_transform(x))
