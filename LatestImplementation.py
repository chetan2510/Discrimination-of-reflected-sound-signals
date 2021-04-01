#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa.display
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pickle
# get_ipython().run_line_magic('matplotlib', 'inline')

# Reading files
print("Reading the excel files .......")
obj_two_data = pd.read_excel('object2.xlsx', header = None)
obj_one_data = pd.read_excel('object1.xlsx', header = None)

# Data rendering and removing unwanted collumns
array_obj_two = np.array(obj_two_data)
array_obj_two = np.delete(array_obj_two, 1, 0)
array_obj_two = np.delete(array_obj_two, 1, 0)
array_obj_one = np.array(obj_one_data)
array_obj_one = np.delete(array_obj_one, 1, 0)
array_obj_one = np.delete(array_obj_one, 1, 0)

print("Values read object one ", array_obj_one.shape)
print("Values read object two", array_obj_two.shape)

## Genrating features for object2
max_values_obj2 = []
sum_obj2 = []
max_value = 0;

for i in range(array_obj_two.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_two[i], NFFT=256, Fs=2, noverlap=0);
    max_value = np.amax(abs(spectrum[0]))
    sum_obj2.append(np.sum(spectrum))
    max_values_obj2.append(max_value)

print("Generated features for object2")

# Genrating features for object1
max_values_obj1 = []
max_value = 0;
sum_obj1 = []

for i in range(array_obj_one.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_one[i], NFFT=256, Fs=2, noverlap=0)
    max_value = np.amax(abs(spectrum[0]))
    sum_obj1.append(np.sum(spectrum))
    max_values_obj1.append(max_value)

print("Generated features for object1")

# Merging and creating data frame
max_freq_dataframe_obj_one = pd.DataFrame(max_values_obj1, columns=["MaxFrequency"])
max_freq_dataframe_obj_two = pd.DataFrame(max_values_obj2, columns=["MaxFrequency"])
sum_obj1 = pd.DataFrame(sum_obj1, columns=["MaxSpectrumSum"])
sum_obj2 = pd.DataFrame(sum_obj2, columns=["MaxSpectrumSum"])
max_freq_dataframe_obj_one["Target"] = 0
max_freq_dataframe_obj_two["Target"] = 1
frames_freq = [max_freq_dataframe_obj_one, max_freq_dataframe_obj_two]
max_sum_frames = [sum_obj1, sum_obj2]
final_freq_frame = pd.concat(frames_freq)
final_max_sum = pd.concat(max_sum_frames)
final_data_frame = pd.concat([final_freq_frame, final_max_sum], axis =1)

# Now splitting the data to test and training sets.
X = final_data_frame.drop("Target", axis =1)
y = final_data_frame["Target"]
np.random.seed(42)

print("Merged and framed the data. Object1 is classified as 0 and Object2 is classified as 1")

# split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # 20 % data to be used for testing

# put models in the dictionary
models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier(),
         "Decission Tree": tree.DecisionTreeClassifier()}

# Create a function to fit and score models
# fits and evaluates the machine learning models
# X_train: training data(no labels)
# X_test: testing data(no labels)
# y_train: training labels
# y_test: testing labels
def fit_and_score(models, X_train, X_test, y_train, y_test):

    # set a random seed
    np.random.seed(42)
    #Male a dictionary to keep model scores
    model_scores = {}
    #loop through models
    for name, model in models.items():
        # fit the mdoel to the data
        model.fit(X_train, y_train)
        # evaluate the mdoel and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

print("Training models")
model_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();

print("Saving random forest model")

# save the model to disk
filename = 'randomforest_model.sav'
pickle.dump(models["Random Forest"], open(filename, 'wb'))

# read_file = pd.read_csv('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_2/frontyellow-0.8/data/002.csv', header = None, usecols=list(range(16384)))
# read_file.to_excel(r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/test_obj1.xlsx', index=None)
# test_data = pd.read_excel('test_obj1.xlsx')
# test_data.shape
#
#
# # In[69]:
#
#
# array_test = np.array(test_data)
# array_test[0]
#
#
# # In[70]:
#
#
# spectrum, freqs, t, im = plt.specgram(array_test[0], NFFT=256, Fs=2, noverlap=0)
# test_max_freq = np.amax(abs(spectrum[0]))
# test_max_sum = np.sum(spectrum)
# # test_max_freq = np.array(test_max_freq)
# # test_max_sum = np.array(test_max_sum)
# # test_max_freq
#
#
# # In[71]:
#
#
# max_freq = pd.DataFrame([test_max_freq], columns=["MaxFrequency"])
# sum_obj = pd.DataFrame([test_max_sum], columns=["MaxSpectrumSum"])
# frames = [max_freq, sum_obj]
# final_data_frame = pd.concat(frames, axis =1)
# final_data_frame
#
#
# # In[72]:
#
#
# value = models["KNN"].predict(final_data_frame)
# print(value)
#
#
# # In[73]:
#
#
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import plot_roc_curve
#
# plot_roc_curve(models["KNN"], X_test, y_test);
#
#
# # In[74]:
#
#
# # Confusion matrix
# import seaborn as sns # build on top pf matplotlib
#
# y_preds = models["KNN"].predict(X_test)
#
# def plot_conf_mat(y_test, y_preds):
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
#     plt.xlabel("True label")
#     plt.ylabel("Predicted label")
#
# plot_conf_mat(y_test, y_preds)
#
#
# # In[75]:
#
#
# # Check best hyperparameters
# knn_model = models["KNN"]
#
#
# # In[76]:
#
#
# # Cross validated recall
# cv_recall = cross_val_score(knn_model, X, y, cv=5, scoring="recall")
#
# cv_recall_mean = np.mean(cv_recall)
# cv_recall_mean
#
#
# # In[77]:
#
#
#
# # corss validated f1-score
# cv_f1_score = cross_val_score(knn_model, X, y, cv=5, scoring="f1")
#
# cv_rf_mean = np.mean(cv_f1_score)
# cv_rf_mean
#
#
# # In[ ]:




