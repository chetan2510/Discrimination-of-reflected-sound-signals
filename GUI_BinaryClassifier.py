import pickle
from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BinaryClassifier:

    # The first function that is called when the code runs
    def __init__(self, win):
        lbl1 = Label(win, text='Choose a file')
        lbl1.place(x=100, y=50)
        self.btn1 = Button(win, text = 'Choose file', command = self.openfile)
        self.btn1.place(x=200, y=50)
        self.t1 = Entry(bg="#eee")
        self.t1.place(x=300, y=50)
        lbl1 = Label(win, text='Choose a Model')
        lbl1.place(x=100, y=150)
        self.btnchoosemodel = Button(win, text='Choose Model', command=self.choosemodel)
        self.btnchoosemodel.place(x=220, y=150)
        self.t2 = Entry(bg="#eee")
        self.t2.place(x=340, y=150)
        self.btnpredict = Button(win, text='Predict', command=self.predict)
        self.btnpredict.place(x=100, y=200)
        self.t3 = Entry(bg="#eee")
        self.t3.place(x=200, y=200, width=100)
        lbl7 = Label(win, text='Number of Scanned Points')
        lbl7.place(x=350, y=200)
        self.t7 = Entry(bg="#eee")
        self.t7.place(x=540, y=200, width=80)
        self.exitButton = Button(win, text='Exit', command=exit)
        self.exitButton.place(x=100, y=250)
        lbl4 = Label(win, text='Enter Row Number')
        lbl4.place(x=100, y=100)
        self.t4 = Entry(bg="#eee")
        self.t4.place(x=220, y=100, width=30)
        lbl5 = Label(win, text='Enter Starting Column Number')
        lbl5.place(x=260, y=100)
        self.t5 = Entry(bg="#eee")
        self.t5.place(x=450, y=100, width=70)
        lbl6 = Label(win, text='Enter ending Column Number')
        lbl6.place(x=540, y=100)
        self.t6 = Entry(bg="#eee")
        self.t6.place(x=750, y=100, width=70)

    # Method to open the file dialog
    def openfile(self):
        self.import_file_path = filedialog.askopenfilename()
        self.t1.insert(END, str(self.import_file_path))

    # Method to choose the model
    def choosemodel(self):
        model = filedialog.askopenfilename()
        self.t2.insert(END, str(model))
        self.loaded_model = pickle.load(open(model, 'rb'))
        print("model choosen successfull")

    # Method to predict the time samples from the CSV files
    def predict(self):
        rowNumber = 0;
        startColNumber = 0;
        endColumnNumber = 16384
        if int(self.t4.get()) >= 0:
            rowNumber = int(self.t4.get())
        if (int(self.t5.get()) > 0 and int(self.t5.get()) <= 16384):
            startColNumber = int(self.t5.get())
        if (int(self.t6.get()) > 0 and int(self.t6.get()) <= 16384):
            endColumnNumber = int(self.t6.get())
        print("RowNumber choose:",rowNumber)
        print("Start CollumnNumber choose:", startColNumber)
        print("End CollumnNumber choose:", endColumnNumber)
        numberOfSamples = len(list(range(startColNumber, endColumnNumber)))
        self.t7.delete(0, 'end')
        self.t7.insert(END, str(numberOfSamples))
        print("Number of samples:", numberOfSamples)
        file = pd.read_csv(self.import_file_path, header = None, usecols=list(range(startColNumber, endColumnNumber)))
        print("file reading successfull")
        array_test = np.array(file)
        spectrum, freqs, t, im = plt.specgram(array_test[rowNumber], NFFT=256, Fs=2, noverlap=0)
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
        plt.show()

window=Tk()
mywin=BinaryClassifier(window)

# Window title
window.title('Sound signal discrimination')

# Window geometry
window.geometry("800x300+10+10")
window.mainloop()