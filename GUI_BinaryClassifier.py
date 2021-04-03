import pickle
from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BinaryClassifier:
    def __init__(self, win):
        lbl1 = Label(win, text='Choose a file')
        lbl1.place(x=100, y=50)
        self.btn1 = Button(win, text = 'Choose file', command = self.openfile)
        self.btn1.place(x=200, y=50)
        self.t1 = Entry()
        self.t1.place(x=300, y=50)
        lbl1 = Label(win, text='Choose a Model')
        lbl1.place(x=100, y=100)
        self.btnchoosemodel = Button(win, text='Choose Model', command=self.choosemodel)
        self.btnchoosemodel.place(x=220, y=100)
        self.t2 = Entry()
        self.t2.place(x=340, y=100)
        self.btnpredict = Button(win, text='Predict', command=self.predict)
        self.btnpredict.place(x=100, y=150)
        self.t3 = Entry()
        self.t3.place(x=200, y=150)
        self.exitButton = Button(win, text='Exit', command=exit)
        self.exitButton.place(x=100, y=200)


    def openfile(self):
        self.import_file_path = filedialog.askopenfilename()
        self.t1.insert(END, str(self.import_file_path))

    def choosemodel(self):
        model = filedialog.askopenfilename()
        self.t2.insert(END, str(model))
        self.loaded_model = pickle.load(open(model, 'rb'))
        print("model choosen successfull")

    def predict(self):
        file = pd.read_csv(self.import_file_path, header = None, usecols=list(range(16384)))
        print("file reading successfull")
        array_test = np.array(file)
        spectrum, freqs, t, im = plt.specgram(array_test[0], NFFT=256, Fs=2, noverlap=0)
        test_max_freq = np.amax(abs(spectrum[0]))
        test_max_sum = np.sum(spectrum)
        max_freq = pd.DataFrame([test_max_freq], columns=["MaxFrequency"])
        sum_obj = pd.DataFrame([test_max_sum], columns=["MaxSpectrumSum"])
        frames = [max_freq, sum_obj]
        final_data_frame = pd.concat(frames, axis=1)
        result = self.loaded_model.predict(final_data_frame)
        final_prediction = "Object-1"
        if result[0] != 0:
            final_prediction = "Object-2"
        self.t3.delete(0, 'end')
        self.t3.insert(END, str(final_prediction))


window=Tk()
mywin=BinaryClassifier(window)
window.title('Sound signal discrimination')
window.geometry("800x300+10+10")
window.mainloop()