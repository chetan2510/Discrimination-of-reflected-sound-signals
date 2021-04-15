import pickle
from tkinter import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.font as tkFont

class BinaryClassifier:

    # The first function that is called when the code runs
    def __init__(self, win):
        fontStyle = tkFont.Font(family="Lucida Grande", size=20)

        # Label on the top
        lbl1 = Label(win, text='Discrimination Of Reflected Sound Signals', fg= "blue", font=fontStyle, background='#ADD8E6')
        lbl1.place(x=200, y=20)

        # to upload a file button
        self.btn1 = Button(win, text = 'Choose file', command = self.openfile, borderwidth=0)
        self.btn1.place(x=200, y=70)
        self.t1 = Entry(bg="#ADD8E6")
        self.t1.place(x=310, y=70)

        lbl4 = Label(win, text='Enter Row Number', fg= "blue", background='#ADD8E6')
        lbl4.place(x=300, y=110)
        self.t4 = Entry(bg="#ADD8E6")
        self.t4.place(x=450, y=110, width=50)

        lbl5 = Label(win, text='Enter Starting Column', fg= "blue", background='#ADD8E6')
        lbl5.place(x=300, y=150)
        self.t5 = Entry(bg="#ADD8E6")
        self.t5.place(x=450, y=150, width=50)

        lbl6 = Label(win, text='Enter Signal Length', fg= "blue", background='#ADD8E6')
        lbl6.place(x=300, y=190)
        self.t6 = Entry(bg="#ADD8E6")
        self.t6.place(x=450, y=190, width=50)

        self.btnchoosemodel = Button(win, text='Choose Model', command=self.choosemodel, borderwidth=0)
        self.btnchoosemodel.place(x=200, y=240)
        self.t2 = Entry(bg="#ADD8E6")
        self.t2.place(x=310, y=240)


        self.btnpredict = Button(win, text='Predict object', command=self.predict, borderwidth=0)
        self.btnpredict.place(x=200, y=310)

        self.t3 = Entry(bg="#ADD8E6")
        self.t3.place(x=310, y=310, width=100)

        lbl7 = Label(win, text='Points Scanned', fg= "blue", background='#ADD8E6')
        lbl7.place(x=200, y=350)
        self.t7 = Entry(bg="#ADD8E6")
        self.t7.place(x=310, y=350, width=80)

        self.exitButton = Button(win, text='Exit', command=exit, borderwidth=0)
        self.exitButton.place(x=320, y=400)

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
            signalLength = int(self.t6.get())
        print("RowNumber choose:",rowNumber)
        print("Start CollumnNumber choose:", startColNumber)
        print("End CollumnNumber choose:", signalLength)
        endColumnNumber = signalLength + startColNumber
        numberOfSamples = len(list(range(startColNumber, endColumnNumber)))
        self.t7.delete(0, 'end')
        self.t7.insert(END, str(numberOfSamples))
        print("Number of samples:", numberOfSamples)
        file = pd.read_excel(self.import_file_path, header = None, usecols=list(range(startColNumber, endColumnNumber)))
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
window.title('Frankfurt University of Applied Sciences')

# Window geometry
window.geometry("700x600+10+10")
window.configure(background='#ADD8E6')
window.mainloop()