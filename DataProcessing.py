#!/usr/bin/env python
# coding: utf-8

# In[54]:


# import os
# import glob
# import pandas as pd

# os.chdir("/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_1/side1-1.6/data")
# file_extension = '.csv'
# all_filenames = [i for i in glob.glob(f"*{file_extension}")]
# all_filenames

# df = pd.read_csv(all_filenames[0])
# print(df.shape)
# combined_csv_data = pd.concat([pd.read_csv(f) for f in all_filenames])
# os.chdir('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals')
# !pwd

# combined_csv_data.to_csv('combined_csv_data_side1-1.6.csv') #Saving our combined csv data as a new file!


# In[69]:


# def process_all_files():
# #     file_names = ["front-black-1", "front-black-1.2", "front-black0.5", "front-yellow-1", "front-yellow0.5", "frontblack-0.8",
# #                  "frontblack-1.5", "frontblack-1.6", "frontyellow-0.8", "frontyellow-1.6"]
#     for file_name in file_names:
#         csv_path = os.path.join("/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/ObjectTwoData", file_name, "data")
#         print(csv_path)
#     os.chdir("/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/ObjectTwoData")
#     file_extension = '.csv'
#     !ls
# #     all_filenames = [i for i in glob.glob(f"*{file_extension}")]
# #     all_filenames
#     df = pd.read_csv(all_filenames[0])
#     print(df.shape)
# #     combined_csv_data = pd.concat([pd.read_csv(f) for f in all_filenames])
# #     os.chdir('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals')
# #     combined_csv_data.to_csv('combined_csv_data_obj_2.csv') #Saving our combined csv data as a new file!
# #     print("Combined successfully")


# In[71]:


process_all_files()


# In[76]:


import glob

import pandas as pd


path =r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_1'

filenames = glob.glob(path + "/*/*/*.csv")

dfs = []

for filename in filenames:

    dfs.append(pd.read_csv(filename, header=None, usecols=list(range(16384))), index = None)

data = pd.concat(dfs)

data.to_excel(r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/ObjectOneData/object1data.xlsx', index=None)
print("Completed")


# In[80]:


read_file = pd.read_csv('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Object_1_Merged.csv', header = None, usecols=list(range(16384)))
read_file.to_excel(r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/object1.xlsx', index=None)


# In[79]:


read_file = pd.read_csv('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Object_2_Merged.csv', header = None, usecols=list(range(16384)))
read_file.to_excel(r'/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/object2.xlsx', index=None)


# In[56]:


list_all_files()


# In[68]:


# from scipy.signal import chirp, spectrogram
# import matplotlib.pyplot as plt
# import numpy as np

# t = np.linspace(0, 10, 1500)
# w = chirp(t, f0=6, f1=1, t1=10, method='linear')
# plt.plot(t, w)
# plt.title("Linear Chirp, f(0)=6, f(10)=1")
# plt.xlabel('t (sec)')
# plt.show()


# In[ ]:


# import csv
# df
# object_one_attributes = {}
# object_one_attributes["Object1"] = x
# object_one = pd.DataFrame(object_one_attributes)
# object_one.tail()
# object_one.head()
# len(object_one)
# plt.plot(x);
# with open('002.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         for sample in row:
#             x.append(sample)

# len(x)
# plt.plot(x);
# import os
# import glob
# import pandas as pd
# os.chdir("/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_1/side1-0.5/data") 

# extension = 'csv'
# all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

# #export to csv
# combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
# y = []

# with open('/Users/brian/Desktop/Discrimination-of-reflected-sound-signals/Files_for_task_2_and_4/Object_1/side1-0.5/data/combined_csv.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         for sample in:
#             y.append(sample)

# len(combined_csv)
# plt.plot(y);

# import csv

# with open('001.csv', newline='') as csvfile:
#     data = list(csv.reader(csvfile))

# print(len(data[0]))
# wines = np.genfromtxt("001.csv", delimiter=";", skip_header=0)
# wines
# data[0][0]
# len(data[0])

# np_array = np.array(data)
# len(np_array[0])

# reshaped_array = np.reshape(np_array, (len(data[0]), 1))
# reshaped_array

# a_dataframe = pd.DataFrame(reshaped_array, columns=["Object"])

