import tkinter as tk
from tkinter import filedialog

def handle_focus_in(_):
    BirdIDText.delete(0, tk.END)
    BirdIDText.config(fg='black')

def handle_focus_out(_):
    BirdIDText.delete(0, tk.END)
    BirdIDText.config(fg='grey')
    BirdIDText.insert(0, "Bird ID")

def handle_enter(txt):
    print(BirdIDText.get())
    handle_focus_out('dummy')

def FileExplorerFunction(click):
    file = filedialog.askdirectory()
    FileDisplay = tk.Label(text=file)
    FileDisplay.grid(row=3, column=1)
    return(file)

def BirdIDButtonClick(BirdID):
    BirdID = BirdIDText.get()
    return(BirdID)

gui = tk.Tk()
greeting = tk.Label(text="Welcome to the AVN gui!")
greeting.grid(row= 0)

BirdIDText = tk.Entry(text="Bird ID", font=("Arial",15), justify="center", fg="grey")
BirdIDText.insert(0, "Bird ID")
BirdIDText.bind("<FocusIn>", handle_focus_in)
BirdIDText.bind("<FocusOut>", handle_focus_out)
#BirdIDText.bind("<Return>", handle_enter)
BirdIDText.grid(row = 1, column=1)

BirdIDLabel = tk.Label(text="Bird ID:")
BirdIDLabel.grid(row=1, column=0)

BirdIDButton = tk.Button(text="Enter")
BirdID = BirdIDButton.bind("<Button-1>", BirdIDButtonClick)
BirdIDButton.grid(row = 2)

FileExplorer = tk.Button(text="Find Folder")
test = FileExplorer.bind("<Button-1>", FileExplorerFunction)
FileExplorer.grid(row = 3, column=0)

MinThresholdText = tk.Entry(justify="center")
MinThresholdText.grid(row= 4, column=1)
MinThresholdLabel = tk.Label(text="Min Threshold:")
MinThresholdLabel.grid(row=4,column=0)

MaxThresholdText = tk.Entry(justify="center")
MaxThresholdText.grid(row= 5, column=1)
MaxThresholdLabel = tk.Label(text="Max Threshold:")
MaxThresholdLabel.grid(row=5,column=0)

gui.mainloop()


