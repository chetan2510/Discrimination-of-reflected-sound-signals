from tkinter import *
import pandas as pd
import numpy as np
import tensorflow as tf

class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='Enter File path:')
        self.lbl1.place(x=100, y=50)
        self.t1=Entry(bd=3)
        self.t1.place(x=200, y=50)        
        
        self.lbl2=Label(win, text='Enter Row number:')
        self.lbl2.place(x=80, y=100)
        self.t2=Entry(bd=3)
        self.t2.place(x=200, y=100)

        self.btn1 = Button(win, text='Check', command=self.predict)
        self.btn1.place(x=200, y=150)
        
        self.lbl3=Label(win, text='Result:')
        self.lbl3.place(x=150, y=200)
        self.t3=Entry()
        self.t3.place(x=200, y=200)
    
    def predict(self):
        self.t3.delete(0, 'end')
        dataframe = pd.read_excel(str(self.t1.get()), header= None)
        data = np.array(dataframe)
        
        normdata = np.zeros((data.shape[0],data.shape[1]))
        for i in range(data.shape[0]):
            normdata[i] = data[i]/max(abs(data[i]))
        
        x_pred = normdata.reshape(normdata.shape[0], normdata.shape[1], 1)
        model = tf.keras.models.load_model("cnn1d.h5")
        y_pred = model.predict(x_pred)
        y_pred = np.argmax(y_pred, axis= 1)
        if (y_pred[int(self.t2.get())] == 0):
            self.t3.insert(END, str('This is Object 1'))
        else:
            self.t3.insert(END, str('This is Object 2'))     

window=Tk()
mywin=MyWindow(window)
window.title('Discrimination of reflected sound signals - CNN')
window.geometry("400x300+10+10")
window.mainloop()