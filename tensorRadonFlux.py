from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.histograms import _ravel_and_check_weights
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#radon data
rad = pd.read_csv("radonKR.csv", sep = ',', index_col = False)
rad['date'] = pd.to_datetime(rad['date'], dayfirst = True, format = "%d.%m.%Y")


#importing the weather data
wth_data = pd.read_csv("weather.csv", sep = ',', index_col = 0)
wth_data['Местное время в Москве (центр, Балчуг)'] = wth_data['Местное время в Москве (центр, Балчуг)'].str.slice(stop=10)
wth_data['Местное время в Москве (центр, Балчуг)'] = pd.to_datetime(wth_data['Местное время в Москве (центр, Балчуг)'], dayfirst=True, format = "%d.%m.%Y")

#preparing the dataset
catcols = ['DD', 'WW']
wth_data = wth_data.sort_values(by='Местное время в Москве (центр, Балчуг)')
wth_data = wth_data.reset_index()
wth_data = wth_data.drop(['index'], axis = 1)
wth_data = wth_data.rename(columns={'Местное время в Москве (центр, Балчуг)':'date'})
wth_data = pd.get_dummies(wth_data, columns = catcols, dummy_na=False)
rad = pd.merge(rad, wth_data, how = 'inner', on = 'date')
rad = rad.drop(['rfd_d'], axis = 1)
timestamps_orig = rad.pop('date')
#onehot encoding
#rad = pd.get_dummies(rad, columns = catcols, dummy_na=False)
#print(pred_set)

#function for collapsing categoricals
def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

#split into test and train sets:
ds_trn = rad.sample(frac = 0.8, random_state = 0)
ds_tst = rad.drop(ds_trn.index)
# print(ds_tst)

#split features from labels, label - value we're looking for
train_features = ds_trn.copy()
test_features = ds_tst.copy()
train_labels = train_features.pop('rfd')
test_labels = test_features.pop('rfd')

# #check out the graphs
# #sns.pairplot(train_dataset[['usv/h', 't', 'p0', 'humid', 'windspd', 'Td', 'rrr', 'mmprecip24', 'mmprecip48']], diag_kind='kde')
# #plt.show()
#normalize values:
#train_features = np.asarray(train_features).astype('float32') 
#ALWAYS CHECK FOR ',' INSTEAD OF '.' IN DATA!
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
first = np.array(train_features[:1])

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [t]')
    plt.legend()
    plt.grid(True)

#define the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 50)

#Full DNN model
dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    train_features, train_labels,
    batch_size = 256,
    validation_split=0.2,
    verbose=0, epochs=500,
    # callbacks = [callback]
    )

# plot_loss(history)
# plt.show()

#Predictions
# predictions = dnn_model.predict(test_features).flatten()
# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, predictions)
# plt.xlabel('True Values [usv/h]')
# plt.ylabel('Predictions [usv/h]')
# plt.show()

#Error distribution for predictions
# error = predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel('Prediction Error [usv/h]')
# _ = plt.ylabel('Count')
# plt.show()

#Show original plot
# y_orig = ds['usv/h']
# plt.plot(timestamps_orig,y_orig)
# plt.gcf().autofmt_xdate()
# plt.xticks(timestamps_orig[::56])
# plt.xlabel("Date")
# plt.ylabel("Dose rate, usv/h")
# plt.title("Actual doserate at mt. Beshtau in 2018-19")
# plt.ylim([0.4, 0.85])
# plt.show()
 
#define the savitzky-golay smoothing algorithm
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError('window_size and order have to be type int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('Window size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('Window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    #precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode = 'valid')

#Make predictions 
ds_pred = wth_data
timestamps = ds_pred.pop('date')
# ds_pred['vconvect'] = ( ds_pred['temp'] - 11.5 ) / 1.7253
#ds_pred = pd.get_dummies(ds_pred, columns = catcols, dummy_na=False)

ds_pred = np.asarray(ds_pred).astype(np.float32)
# print(ds_pred)
# print(hs_pred)
# timestamps = timestamps.str.slice(0, -9, 1)
# timestamps_orig = timestamps_orig.str.slice(0, -9, 1)

predix = dnn_model.predict(ds_pred).flatten()
print(predix)
predix = pd.DataFrame(predix)
Q1 = predix.quantile(0.05)
Q3 = predix.quantile(0.95)
IQR = Q3 - Q1

outliers = predix[((predix < (Q1 - 1.5 * IQR)) |(predix > (Q3 + 1.5 * IQR))).any(axis=1)]
for i in outliers.index:
    predix.loc[i] = ((predix.loc[i-1] + predix.loc[i+1])/2)

#predix.to_csv('predix.csv')
predix = predix[0].tolist()
predix = np.asarray(predix)
pred_smoothed = savitzky_golay(predix, 15, 2)
output1 = pd.DataFrame(predix, index=timestamps, columns = ['prediction'])
timestamps = timestamps[:-1]
output2 = pd.DataFrame(pred_smoothed, index=timestamps, columns = ['smoothed_p'])

outs = [output1, output2]
output = pd.concat(outs, 1)
output.to_csv('outputcat.csv')
wth_data.to_csv('outputweather.csv')
#plt.plot(timestamps,predix, color = 'red')
# plt.plot(timestamps,predix, color = 'turquoise')
plt.plot(timestamps, pred_smoothed, color = 'blue')
#original plot
y_orig = rad['rfd']
plt.plot(timestamps_orig,y_orig, color = 'cyan')
##############
plt.gcf().autofmt_xdate()
plt.xticks(timestamps[::112])
plt.xlabel("Date")
plt.ylabel("RFD")
plt.title('Predicted RFD')
plt.savefig('prediction.svg', format = 'svg', dpi = 1200)
plt.ylim([0, 200])
plt.show()
# #sns.pairplot(raw_dataset, kind = 'reg', plot_kws = {'line_kws':{'color':'blue'}})
# #plt.show()
