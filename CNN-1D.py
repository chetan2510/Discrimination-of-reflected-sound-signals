import pandas as pd
import numpy as np
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataframe1 = pd.read_excel("Data/Object1.xlsx", header= None)
data1 = np.array(dataframe1)

dataframe2 = pd.read_excel("Data/Object2.xlsx", header= None)
data2 = np.array(dataframe2)

normdata1 = np.zeros((data1.shape[0],data1.shape[1]))
for i in range(data1.shape[0]):
    normdata1[i] = data1[i]/max(abs(data1[i]))

normdata2 = np.zeros((data2.shape[0],data2.shape[1]))
for i in range(data2.shape[0]):
    normdata2[i] = data2[i]/max(abs(data2[i]))

num_classes = 2

X = np.vstack((normdata1,normdata2)) # Features
Y = np.hstack((np.zeros(normdata1.shape[0]),np.ones(normdata2.shape[0]))) # Labels

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = models.Sequential()
model.add(layers.Input(shape= x_train.shape[1:]))
model.add(layers.Conv1D(128, 7, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 6))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(0.001), metrics=['accuracy'])

model.summary()

# Train the model
print('Training the model:')
model.fit(x_train, y_train, epochs= 10, validation_split= 0.2, verbose= 1)

# Test the model after training
print('Testing the model:')
model.evaluate(x_test, y_test, verbose= 1)

# Confusion Matrix
y_actual = y_test
y_est = model.predict(x_test)
y_est = np.argmax(y_est, axis= 1)
cm = confusion_matrix(y_actual, y_est).ravel()
tn, fp, fn, tp  = cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm.reshape(2,2))
disp.plot()

tpr = tp/(tp + fn)
tnr = tn/(tn + fp)
fdr = fp/(fp + tp)
npv = tn/(tn + fn)

# Save the Model
model.save('cnn1d.h5')