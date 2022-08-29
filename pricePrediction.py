import os, sys, pandas as pd, numpy as np, utils
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    raise Exception("file is not provided")

# transform data_file contents into dataframe
df = pd.read_csv(data_file, index_col='Date', usecols=['Date','Close'], parse_dates=True)
dataset_orig = df.to_numpy() # extract the underlying data in terms of ndarray with shape (<#ofsamples>,1)
scale = MinMaxScaler(feature_range=(0, 1))
scaled_dataset_price = scale.fit_transform(dataset_orig)

# reorganize data such that every row contains some number of data prices
window_size = 31 #number that includes both data features and class, i.e. the first window_size-1 are feature columns and last one is class
data_price = utils.reorganize(scaled_dataset_price, window_size)

#data setup for training/testing
test_size=5
trainX = data_price[:-test_size,:-1]
trainX = np.reshape( trainX, (trainX.shape[0],trainX.shape[1],1) )
trainY = data_price[:-test_size,-1]
testX = data_price[-test_size:,:-1]
testX = np.reshape( testX, (testX.shape[0],testX.shape[1],1) )
testY = dataset_orig[-test_size : ]

model = Sequential()
model.add(LSTM(units=50,return_sequences=True))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(trainX,trainY, batch_size=1, epochs=20, verbose=0)

predictions = model.predict(testX)
predictions = scale.inverse_transform(predictions)

model_error = np.sqrt( np.mean( (predictions - testY)**2 ) )
print("model error:",model_error)

test_df = df.iloc[-test_size:]
test_df['predictions'] = predictions
print(test_df)

april4data = data_price[-1,1:].reshape((1,-1,1))
april4predict = model.predict(april4data)
april4predict = scale.inverse_transform(april4predict)[0,0]
print(april4predict)
with open('price_predictions.txt','a') as fhandler:
    fhandler.write( os.path.basename(data_file).split('.')[0] + ": " + repr(april4predict) + '\n')
