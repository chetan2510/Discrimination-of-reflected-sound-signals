import pickle
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.font as tkFont
import time
import pickle

class BinaryClassifier:

    # The first function that is called when the code runs
    def __init__(self, win):
        fontStyle = tkFont.Font(family="Georgia", size=20)
        fontStyleStep = tkFont.Font(family="Times", size=10, weight="bold")

        # Adding the progress bar
        self.my_progress = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        self.my_progress.pack(pady=20)
        self.my_progress.place()
        self.my_progress.place(x=100, y=150, width=500)

        # Adding the second progress bar
        self.my_progress2 = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        self.my_progress2.pack(pady=20)
        self.my_progress2.place()
        self.my_progress2.place(x=100, y=360, width=500)

        # Adding the third progress bar
        self.my_progress3 = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        self.my_progress3.pack(pady=20)
        self.my_progress3.place()
        self.my_progress3.place(x=250, y=450, width=230)

        # Label on the top
        lbl1 = Label(win, text='Discrimination Of Reflected Sound Signals', fg= "black", font=fontStyle, background='#A3A3A3')
        lbl1.place(x=100, y=20)

        # To add Step1 indication
        lbl10 = Label(win, text='Step 1) ------------------------------------------------------------------------------------------------------------------', fg="black", font=fontStyleStep, background='#A3A3A3')
        lbl10.place(x=100, y=90)

        # to upload a file button
        lbl22 = Label(win, text='*Choose Input Test Signal', fg="black", background='#A3A3A3')
        lbl22.place(x=100, y=120)
        self.btn1 = Button(win, text = 'Browse', command = self.openfile)
        self.btn1.place(x=550, y=120)
        self.t1 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t1.focus()
        self.t1.place(x=250, y=120, width=280)

        # To add Step2 indication
        lbl11 = Label(win, text='Step 2) ------------------------------------------------------------------------------------------------------------------', fg="black", font=fontStyleStep, background='#A3A3A3')
        lbl11.place(x=100, y=180)

        lbl4 = Label(win, text='*Enter Row Number', fg= "black", background='#A3A3A3')
        lbl4.place(x=100, y=210)
        self.t4 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t4.place(x=250, y=210, width=280)

        lbl5 = Label(win, text='*Enter Starting Column', fg= "black", background='#A3A3A3')
        lbl5.place(x=100, y=240)
        self.t5 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t5.place(x=250, y=240, width=280)

        lbl6 = Label(win, text='*Enter Signal Length', fg= "black", background='#A3A3A3')
        lbl6.place(x=100, y=270)
        self.t6 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t6.place(x=250, y=270, width=280)

        # To add Step3 indication
        lbl13 = Label(win, text='Step 3) ------------------------------------------------------------------------------------------------------------------', fg="black", font=fontStyleStep, background='#A3A3A3')
        lbl13.place(x=100, y=300)

        #self.btnchoosemodel = Button(win, text='Choose Model', command=self.choosemodel, borderwidth=0)
        #self.btnchoosemodel.place(x=200, y=240)
        #self.t2 = Entry(bg="#ADD8E6")
        #self.t2.place(x=310, y=240)
        #self.var1 = IntVar()
        #self.check = Checkbutton(window, text='Select RF Classification Model', variable=self.var1, onvalue=1, offvalue=0, command=self.selection)
        #self.check.place(x=310, y=240)

        lbl15 = Label(win, text='*Choose Trained Model', fg="black", background='#A3A3A3')
        lbl15.place(x=100, y=330)
        self.btnchoosemodel = Button(win, text='Browse', command=self.choosemodel)
        self.btnchoosemodel.place(x=550, y=330)
        self.t2 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t2.place(x=250, y=330, width=280)

        # To add Step4 indication
        lbl16 = Label(win, text='Step 4) ------------------------------------------------------------------------------------------------------------------', fg="black", font=fontStyleStep, background='#A3A3A3')
        lbl16.place(x=100, y=390)

        #lbl17 = Label(win, text='Press to Predict', fg= "black", background='#ADD8E6')
        #lbl17.place(x=100, y=420)

        self.btnpredict = Button(win, text='Predict Object', command=self.predict)
        self.btnpredict.place(x=250, y=420, width=230)

        lbl17 = Label(win, text='Predicted Output', fg= "black", font=fontStyleStep, background='#A3A3A3')
        lbl17.place(x=250, y=490)
        self.t3 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t3.place(x=380, y=490, width=100)

        lbl7 = Label(win, text='# of Scanned Points', fg= "black", font=fontStyleStep, background='#A3A3A3')
        lbl7.place(x=250, y=510)
        self.t7 = Entry(bg="#C0C0C0", highlightthickness=2, highlightcolor='lawn green')
        self.t7.place(x=380, y=510, width=100)

        self.popup = Button(win, text='Show Model Assessment', command=self.popupmsg)
        self.popup.place(x=250, y=550, width=230)

        self.resetButton = Button(win, text='Try Again?', command=self.reset)
        self.resetButton.place(x=270, y=590, width=70)
        self.exitButton = Button(win, text='Exit', command=exit)
        self.exitButton.place(x=400, y=590, width=70)

    # Method to open the file dialog
    def openfile(self):
        self.my_progress['value'] = 0
        self.t1.delete(0, 'end')
        self.import_file_path = filedialog.askopenfilename()
        for x in range(5):
            self.my_progress['value'] +=20
            self.my_progress.update()
            time.sleep(1)

        self.t1.insert(END, str(self.import_file_path))
        #self.btnchoosemodel.pack()
        #self.btnchoosemodel.place(x=200, y=240)

    # Method to choose the model
    def choosemodel(self):
        self.my_progress2['value'] = 0
        self.t2.delete(0, 'end')
        model = filedialog.askopenfilename()
        for x in range(5):
            self.my_progress2['value'] +=20
            self.my_progress2.update()
            time.sleep(1)

        self.t2.insert(END, str(model))
        self.loaded_model = pickle.load(open(model, 'rb'))
        print("model choosen successfull")

    # Method to reset the widgets on the screen
    def reset(self):
        self.t1.delete(0, 'end')
        self.t2.delete(0, 'end')
        self.t3.delete(0, 'end')
        self.t4.delete(0, 'end')
        self.t5.delete(0, 'end')
        self.t6.delete(0, 'end')
        self.t7.delete(0, 'end')
        self.my_progress['value'] = 0
        self.my_progress2['value'] = 0
        self.my_progress3['value'] = 0

    # def selection(self):
    #     if (self.var1.get() == 1):
    #         model = 'randomforest_model_final.sav'
    #         self.loaded_model = pickle.load(open(model, 'rb'))

    # Method to predict the time samples from the excel files
    def predict(self):

        self.my_progress3['value'] = 20
        self.my_progress3.update()
        # STEP 1. To set the initialised values to 0
        self.my_progress3['value'] = 0
        self.t3.delete(0, 'end')
        self.t7.delete(0, 'end')
        rowNumber = 0;
        startColNumber = 0;

        # STEP 2. To read the values from the input fields
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
        if endColumnNumber > 16384:
            endColumnNumber = 16384
        numberOfSamples = len(list(range(startColNumber, endColumnNumber)))
        print("Number of samples:", numberOfSamples)

        # STEP 3. To read the selected excel file
        file = pd.read_csv(self.import_file_path, header = None, usecols=list(range(startColNumber, endColumnNumber)))
        print("file reading successfully")
        array_test = np.array(file)

        # STEP 4. To generate the specgtogram
        self.spectrum, self.freqs, self.t, im = plt.specgram(array_test[rowNumber], NFFT=256, Fs=2, noverlap=0)

        # STEP 5. To extract the features
        test_max_sum = np.sum(self.spectrum)
        test_instantaneous_power = (np.amax(abs(self.spectrum[0]))) / self.t[np.argmax(abs(self.spectrum[0]))]
        test_max_value = np.amax(abs(self.spectrum[0]))
        average_energy = np.sum(self.spectrum) / len(self.spectrum)

        # Calculating Instantaneous Frequency
        fre_spec_sum = 0.0
        # Applying loop on the length of the spectrum
        for i in range(0, len(self.spectrum)):
            fre_spec_sum = fre_spec_sum + np.sum(self.spectrum[i]) * self.freqs[i]
        instantaneous_frequency =  fre_spec_sum / np.sum(self.spectrum)

        # Calculating TFA Group Delay
        time_spec_sum = 0.0
        # Applying loop on the length of the spectrum
        for i in range(0, len(self.t)):
            j = i * 2
            time_spec_sum = time_spec_sum + np.sum(self.spectrum[j]) * self.t[i]
        tfa_Group_Delay = time_spec_sum / np.sum(self.spectrum)

        # STEP 6. To create a data frame
        max_freq_df = pd.DataFrame([test_max_value], columns=["MaxFrequency"])
        spectrum_energy_df = pd.DataFrame([test_max_sum], columns=["Spectrum Energy"])
        instantaneous_power_df = pd.DataFrame([test_instantaneous_power], columns=["Instantaneous Power"])
        average_energy_df = pd.DataFrame([average_energy], columns=["Average Energy"])
        instantaneous_freq_df = pd.DataFrame([instantaneous_frequency], columns=["Instantaneous Frequency"])
        TFA_GroupDelay_df = pd.DataFrame([tfa_Group_Delay], columns=["TFA Group Delay"])
        frames = [max_freq_df, spectrum_energy_df, instantaneous_power_df, average_energy_df, instantaneous_freq_df,
                  TFA_GroupDelay_df]
        final_data_frame = pd.concat(frames, axis=1)

        # STEP 7. To predict using the loaded random forest model
        result = self.loaded_model.predict(final_data_frame)
        final_prediction = "Object-1"
        if result[0] != 0:
            final_prediction = "Object-2"

        for x in range(5):
            self.my_progress3['value'] +=20
            self.my_progress3.update()
            time.sleep(1)

        # STEP 8. To display the results
        self.t3.insert(END, str(final_prediction))
        self.t7.insert(END, str(numberOfSamples))
        plt.show()

    # This method shows the model evaluation in the new window
    def popupmsg(msg):
        fontstylenew = tkFont.Font(family="Georgia", size=10)
        popup = Tk()
        popup.wm_title("Model Evaluation Metrics")
        label1 = ttk.Label(popup, text=' False Discovery Rate (FDR) = 3.9%', font=fontstylenew, background='bisque')
        label2 = ttk.Label(popup, text=' Negative Predictive Value (NPV) = 95.45%', font=fontstylenew, background='bisque')
        label3 = ttk.Label(popup, text=' True Positive Rate (TPR) = 96.05%', font=fontstylenew, background='bisque')
        label4 = ttk.Label(popup, text=' True Negative Rate (TNR) = 95.45%', font=fontstylenew, background='bisque')
        label5 = ttk.Label(popup, text=' Accuracy = 95.77%', font=fontstylenew, background='bisque')
        label6 = ttk.Label(popup, text=' Precision = 96.05%', font=fontstylenew, background='bisque')
        label7 = ttk.Label(popup, text=' Recall = 96.05%', font=fontstylenew, background='bisque')
        label8 = ttk.Label(popup, text=' F1 score = 96.05%', font=fontstylenew, background='bisque')
        label1.pack(side="top", fill="x", pady=10)
        label2.pack(side="top", fill="x", pady=10)
        label3.pack(side="top", fill="x", pady=10)
        label4.pack(side="top", fill="x", pady=10)
        label5.pack(side="top", fill="x", pady=10)
        label6.pack(side="top", fill="x", pady=10)
        label7.pack(side="top", fill="x", pady=10)
        label8.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
        B1.pack()
        popup.mainloop()

window=Tk()
mywin=BinaryClassifier(window)

# Window title
window.title('Welcome to Binary Classification App')

#Logo
#image1 = Image.open("logo.png")
test =PhotoImage(file='logo.png')
label1 = Label(window, width=220, height=100, bg='#A3A3A3', image=test)
#label1.image = test
label1.place(x=510, y=600)

# Window geometry
window.geometry("750x700")
window.configure(background='#A3A3A3')
window.mainloop()