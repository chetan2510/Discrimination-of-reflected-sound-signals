#Python program to create
from tkinter import *
import tkinter as tk


# Function for opening the
# file explorer window
def browseFiles():

    # export_file_path = filedialog.asksaveasfilename(defaultextension='.xlsx')
    # read_file.to_excel(export_file_path, index=None, header=True)
    # test_file = pd.read_excel(import_file_path)
    # print(test_file.shape)

    print(result)
    # laebel.config(text=str(result))

# Create the root window
window = Tk()
  

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