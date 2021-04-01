import pandas as pd
import numpy as np
import tensorflow as tf

dataframe = pd.read_excel("Data/Object1.xlsx", header= None)
data = np.array(dataframe)

normdata = np.zeros((data.shape[0],data.shape[1]))
for i in range(data.shape[0]):
    normdata[i] = data[i]/max(abs(data[i]))

x_pred = normdata.reshape(normdata.shape[0], normdata.shape[1], 1)
model = tf.keras.models.load_model("cnn1d.h5")
y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis= 1)
