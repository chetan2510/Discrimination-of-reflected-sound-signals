#Python program to create
from tkinter import *
from tkinter import filedialog as fd, filedialog
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import tkinter as tk


# Function for opening the
# file explorer window
def browseFiles():
    global read_file
    import_file_path = filedialog.askopenfilename()
    read_file = pd.read_csv(import_file_path, header = None, usecols=list(range(16384)))
    # export_file_path = filedialog.asksaveasfilename(defaultextension='.xlsx')
    # read_file.to_excel(export_file_path, index=None, header=True)
    # test_file = pd.read_excel(import_file_path)
    # print(test_file.shape)
    array_test = np.array(read_file)
    spectrum, freqs, t, im = plt.specgram(array_test[0], NFFT=256, Fs=2, noverlap=0)
    test_max_freq = np.amax(abs(spectrum[0]))
    test_max_sum = np.sum(spectrum)
    max_freq = pd.DataFrame([test_max_freq], columns=["MaxFrequency"])
    sum_obj = pd.DataFrame([test_max_sum], columns=["MaxSpectrumSum"])
    frames = [max_freq, sum_obj]
    final_data_frame = pd.concat(frames, axis=1)
    import_model_path = filedialog.askopenfilename()
    loaded_model = pickle.load(open(import_model_path, 'rb'))
    result = loaded_model.predict(final_data_frame)
    print(result)
    # laebel.config(text=str(result))

# Create the root window
window = Tk()
  
# Set window title
window.title('Discrimination of reflected sound sognals using machine learning')
  
# Set window size
window.geometry("400x400")
  
# #Set window background color
# window.config(background = "white")
  
# Create a File Explorer label
label_file_explorer = Label(window,
                            text = "File Explorer using Tkinter",
                            width = 100, height = 4,
                            fg = "blue")

laebel = tk.Label(window, fg="green")
laebel.pack()

button_explore = Button(window,
                        text = "Browse Files",
                        command = browseFiles)
  
button_exit = Button(window,
                     text = "Exit",
                     command = exit)


# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column = 1, row = 1)
  
button_explore.grid(column = 1, row = 2)
  
button_exit.grid(column = 1,row = 3)
  
# Let the window wait for any events
window.mainloop()