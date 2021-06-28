import glob

import pandas as pd

# Path where the csv files resides
path =r'/Users/Project2/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_1'

# Using glob to retrieve all the csv files
filenames = glob.glob(path + "/*/*/*.csv")

# Creating an array where the files can be appended
dfs = []

# Loop to scan all the files
for filename in filenames:
    dfs.append(pd.read_csv(filename, header=None, usecols=list(range(16384))), index = None)

# Concatenating all the csv files together
data = pd.concat(dfs)

# Converting them to excel
data.to_excel(r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/ObjectOneData/object1data.xlsx', index=None)

# Will print completed once the processing is completed
print("Completed")

