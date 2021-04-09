import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pickle

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

print("Scanning of objects completed")

# Generating features for object2
print("Generating features for object2")
max_values_obj2 = []
sum_obj2 = []
max_value = 0;
for i in range(array_obj_two.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_two[i], NFFT=256, Fs=2, noverlap=0);
    max_value = np.amax(abs(spectrum[0]))
    sum_obj2.append(np.sum(spectrum))
    max_values_obj2.append(max_value)

print("Generated features for object2")

# Generating features for object1
print("Generating features for object1")
max_values_obj1 = []
max_value = 0
sum_obj1 = []

for i in range(array_obj_one.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_one[i], NFFT=256, Fs=2, noverlap=0)
    max_value = np.amax(abs(spectrum[0]))
    sum_obj1.append(np.sum(spectrum))
    max_values_obj1.append(max_value)

print("Generated features for object1")

# Merging and creating data frame
print("Creating data frames for object1 and object2")
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
print("Data frames creation completed for object1 and object2")

# Now splitting the data to testing and training sets.
print("Creating test and training data sets")
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
# X_train: training data
# X_test: testing data
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

print("Training models.....")
model_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar()

print("Saving random forest model")

# Save the model to disk
filename = 'randomforest_model_final.sav'
pickle.dump(models["Random Forest"], open(filename, 'wb'))

# Extracting features TP, TN, FP, FN, FDR, NPV, TPR, TNR, F1, ROC

# Generating confusion matrix
y_actual = y_test
y_est = models["Random Forest"].predict(X_test)
cm = confusion_matrix(y_actual, y_est).ravel()

# Generating TP, TN, FP and FN
tn, fp, fn, tp  = cm
disp = ConfusionMatrixDisplay(confusion_matrix=cm.reshape(2,2))
disp.plot()

# Generating FDR, NPV, TPR, TNR
tpr = tp/(tp + fn)
tnr = tn/(tn + fp)
fdr = fp/(fp + tp)
npv = tn/(tn + fn)
accuracy = (tp+tn)/(tp + tn + fp + fn)
precision = tp/(tp + fp)
recall = tp/(tp + fn)
f1_score = 2 * (precision * recall)/(recall + precision)

# Printing the values TPR, TNR, FDR, NPV, ACCURACY, PRECISION, RECALL and F1 score
print("TPR is :"+str(tpr))
print("TNR is:"+str(tnr))
print("FDR is:"+str(fdr))
print("NPV is:"+str(fdr))
print("Accuracy is:"+str(accuracy))
print("Precision is:"+str(precision))
print("Recall is:"+str(recall))
print("F1 score is:"+str(f1_score))

# Plotting ROC curve
plot_roc_curve(models["Random Forest"], X_test, y_test)
plt.show()






