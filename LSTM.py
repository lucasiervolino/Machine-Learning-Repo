import numpy as np
import pandas as pd

#loads datasets
X = pd.read_csv(path_X)
y = pd.read_csv(path_y)

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
entireData3 = entireData2[16:,:,:]
X_entireData = entireData2[:,:-1,:]
Y_entireData = entireData2[:,1:,:]

seed = 9

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, TimeDistributed

# design network
model = Sequential()
model.reset_states()

model.add(LSTM(200,input_shape=(X_entireData.shape[1],X_entireData.shape[2]), return_sequences=True)) #, stateful=True, )) #dropout=0.1, recurrent_dropout=0.1,
model.add(LSTM(200,input_shape=(X_entireData.shape[1],X_entireData.shape[2]), return_sequences=True))
#model.add(LSTM(100,input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
#model.add(LSTM(23,input_shape=(X_train.shape[1],Y_train.shape[2]),activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
#model.add(LSTM(23,input_shape=(X_train.shape[1],Y_train.shape[2]),activation='relu', dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add((Dense(7, activation='softmax')))

from keras.optimizers import Adam, RMSprop
adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None,  amsgrad=False)
rms = RMSprop(lr=0.006)

model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])


from keras.callbacks import EarlyStopping, ModelCheckpoint

earlystopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

history = model.fit(X_entireData[16:,:,:], Y_entireData[16:,:,:], epochs=200, verbose=2, batch_size=32)


#second predictive model
new_model = Sequential()
batch_size=1
new_model.add(LSTM(200,batch_input_shape=(32,None,1), stateful=True, return_sequences=True)) #, stateful=True, )) #dropout=0.1, recurrent_dropout=0.1,
new_model.add(LSTM(200,batch_input_shape=(32,None,1), stateful=True)) #, stateful=True, )) #dropout=0.1, recurrent_dropout=0.1,
#new_model.add(LSTM(300,input_shape=(None,1), return_sequences=True, stateful=True))
new_model.add(Dense(7))
new_model.compile(loss='sparse_categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
new_model.reset_states()
new_model.set_weights(model.get_weights())
predictions = new_model.predict(entireData3)
predictions = np.argmax(predictions, axis=1)

futureElement = predictions.reshape(len(predictions),1,1)

futureElements = []
futureElements.append(futureElement)

for i in range(4):
    
    futureElement = new_model.predict(futureElement)
    futureElement = np.argmax(futureElement, axis=1)
    futureElement = futureElement.reshape(len(futureElement),1,1)
    futureElements.append(futureElement)

#creates dataframe structure with correct column labels
question_final = pd.DataFrame()
for i in range(len(futureElements)):
    Q = 'Q%i' % (i+41)
    a = np.array(futureElements[i].reshape(len(futureElement)))
    question_final[Q] = a
    
question_final2 = question_final.copy(deep=True)
question_final = question_final.apply(lambda x: dcat[x.name].inverse_transform(x)) #inverts LabelEncoder dict
