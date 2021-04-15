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
import math

##############################################
# STEP 1
# To read all the files from the excel and club them together as a data fram
# NOTE: Keeping header none to avoid the headers (0,1,2,3,4,......)
# Reading files
##############################################

print("Reading the excel files .......")
obj_two_data = pd.read_excel('object2.xlsx', header = None)
obj_one_data = pd.read_excel('object1.xlsx', header = None)

# Data rendering and removing unwanted collumns
# NOTE: This functionality is specifically for MAC and is required to run remove the unwanted rows.
array_obj_two = np.array(obj_two_data)
array_obj_two = np.delete(array_obj_two, 1, 0)
array_obj_two = np.delete(array_obj_two, 1, 0)
array_obj_one = np.array(obj_one_data)
array_obj_one = np.delete(array_obj_one, 1, 0)
array_obj_one = np.delete(array_obj_one, 1, 0)

print("Scanning of objects completed")

# Function to calculate the instantaneous frequency
def calculate_instantaneous_frequency(spectrum, f):
    fre_spec_sum = 0.0
    for i in range(0, len(spectrum)):
        fre_spec_sum = fre_spec_sum + np.sum(spectrum[i]) * f[i]
    return fre_spec_sum / np.sum(spectrum)

# Function to calculate the time frequency group delay
def calculate_TFA_GroupDelay(spectrum, t):
    time_spec_sum = 0.0
    for i in range(0, len(t)):
        j = i * 2
        time_spec_sum = time_spec_sum + np.sum(spectrum[j]) * t[i]
    return time_spec_sum / np.sum(spectrum)


##############################################
# STEP 2
# Here the features for the object 1 and object are generated
# Six features have been extracted from the spectogram namely:
# a) Max value in the spectrum
# a) Instantaneous power
# a) Average Energy
# a) Instantaneous frequency
# a) TFA group delay
# NOTE: These features were choosen after lot of variations. using this features an accuracy of 95.77% has been achieved
##############################################

# Generating features for object1
print("Generating features for object1")
max__spectrum_values_obj1 = [] # Stores the maximum values
spectrum_energy_obj1 = [] # Stores the spectrum energy
instantaneous_power_values_obj1 = [] # Stores the instantaneous power
average_energy_obj1 = [] # Stores the average energy
instantaneous_frequency_array_object1 = [] # Stores the instantaneous frequency
tfa_group_delay_object1 = [] # Stores the TFA group delay

# Here iteration operation is being performed to extract the features from each row of the signal
for i in range(array_obj_one.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_one[i], NFFT=256, Fs=2, noverlap=0)
    instantaneous_power = (np.amax(abs(spectrum[0])))/t[np.argmax(abs(spectrum[0]))]
    max_value = np.amax(abs(spectrum[0]))
    spectrum_energy_obj1.append(np.sum(spectrum))
    max__spectrum_values_obj1.append(max_value)
    average_energy_obj1.append(np.sum(spectrum)/len(spectrum))
    instantaneous_power_values_obj1.append(instantaneous_power)
    instantaneous_frequency_array_object1.append(calculate_instantaneous_frequency(spectrum,freqs))
    tfa_group_delay_object1.append(calculate_TFA_GroupDelay(spectrum, t))

print("Generated features for object1")

# Generating features for object2
print("Generating features for object2")
max_spectrum_values_obj2 = [] # Stores the maximum values
spectrum_energy_obj2 = [] # Stores the spectrum energy
instantaneous_power_values_obj2 = [] # Stores the instantaneous power
average_energy_obj2 = [] # Stores the average energy
instantaneous_frequency_array_object2 = [] # Stores the instantaneous frequency
tfa_group_delay_object2 = [] # Stores the TFA group delay

# Here iteration operation is being performed to extract the features from each row of the signal
for i in range(array_obj_two.shape[0]):
    spectrum, freqs, t, im = plt.specgram(array_obj_two[i], NFFT=256, Fs=2, noverlap=0);
    max_value = np.amax(abs(spectrum[0]))
    instantaneous_power = (np.amax(abs(spectrum[0])))/t[np.argmax(abs(spectrum[0]))]
    spectrum_energy_obj2.append(np.sum(spectrum))
    average_energy_obj2.append(np.sum(spectrum)/len(spectrum))
    max_spectrum_values_obj2.append(max_value)
    instantaneous_power_values_obj2.append(instantaneous_power)
    instantaneous_frequency_array_object2.append(calculate_instantaneous_frequency(spectrum,freqs))
    tfa_group_delay_object2.append(calculate_TFA_GroupDelay(spectrum, t))

print("Generated features for object2")

##############################################
# STEP 3
# Here the data frame is being created. In a data frame porper collumns will be defined and the target value will be set
# Target 0 means its object1 and Target 1 means its object 2
##############################################

# Merging and creating data frame
print("Creating data frames for object1 and object2")
max_freq_dataframe_obj_one = pd.DataFrame(max__spectrum_values_obj1, columns=["MaxFrequency"])
max_freq_dataframe_obj_two = pd.DataFrame(max_spectrum_values_obj2, columns=["MaxFrequency"])
sum_obj1 = pd.DataFrame(spectrum_energy_obj1, columns=["Spectrum Energy"])
sum_obj2 = pd.DataFrame(spectrum_energy_obj2, columns=["Spectrum Energy"])
instantaneous_power_obj1 = pd.DataFrame(instantaneous_power_values_obj1, columns=["Instantaneous Power"])
instantaneous_power_obj2 = pd.DataFrame(instantaneous_power_values_obj2, columns=["Instantaneous Power"])
average_energy_obj1 = pd.DataFrame(average_energy_obj1, columns=["Average Energy"])
average_energy_obj2 = pd.DataFrame(average_energy_obj2, columns=["Average Energy"])
instantaneous_freq_obj1 = pd.DataFrame(instantaneous_frequency_array_object1, columns=["Instantaneous Frequency"])
instantaneous_freq_obj2 = pd.DataFrame(instantaneous_frequency_array_object2, columns=["Instantaneous Frequency"])
TFA_GroupDelay_obj1 = pd.DataFrame(tfa_group_delay_object1, columns=["TFA Group Delay"])
TFA_GroupDelay_obj2 = pd.DataFrame(tfa_group_delay_object2, columns=["TFA Group Delay"])
max_freq_dataframe_obj_one["Target"] = 0
max_freq_dataframe_obj_two["Target"] = 1

##############################################
# STEP 4
# Merging will be performed for all the data frames created right above.
# All the data frames will be merged into on3 data frame named as final_data_frame
##############################################

frames_freq = [max_freq_dataframe_obj_one, max_freq_dataframe_obj_two]
max_sum_frames = [sum_obj1, sum_obj2]
average_energy = [average_energy_obj1, average_energy_obj2]
instantaneous_power = [instantaneous_power_obj1, instantaneous_power_obj2]
instantaneous_freq = [instantaneous_freq_obj1, instantaneous_freq_obj2]
group_delay = [TFA_GroupDelay_obj1, TFA_GroupDelay_obj2]
final_freq_frame = pd.concat(frames_freq)
final_max_sum = pd.concat(max_sum_frames)
final_instantaneous_power = pd.concat(instantaneous_power)
final_average_energy = pd.concat(average_energy)
final_instantaneous_freq = pd.concat(instantaneous_freq)
final_group_delay = pd.concat(group_delay)
data_frames_list = [final_freq_frame, final_max_sum, final_instantaneous_power, final_average_energy, final_instantaneous_freq, final_group_delay]
print("Merged and framed the data. Object1 is classified as 0 and Object2 is classified as 1")
final_data_frame = pd.concat(data_frames_list, axis =1)

##############################################
# STEP 5
# Splitting and performing the test and training on different models
# Going one step forward we took 4 different model to get the best out of them
# Results shows that the random forest gives the highest accuracy
##############################################

# Splitting the data to testing and training sets.
print("Creating test and training data sets")
X = final_data_frame.drop("Target", axis =1)
y = final_data_frame["Target"]
np.random.seed(42)

# split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) # 25 % data to be used for testing

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
plt.show()
print("Training completed!!")
print("Saving random forest model")

##############################################
# STEP 6.
# After training best model choosen is random forest with 95.77% accuracy
# After selecting the model will be saved in the disk memory for future usage
##############################################
filename = 'random_forest_model.sav'
pickle.dump(models["Random Forest"], open(filename, 'wb'))

##############################################
# STEP 7.
# Extracting features TP, TN, FP, FN, FDR, NPV, TPR, TNR, F1, ROC and confusion matrix
##############################################

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