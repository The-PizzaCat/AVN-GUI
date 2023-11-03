try:
    import tkinter as tk
    from tkinter import filedialog
    import avn
    import csv
    import avn.dataloading
    import avn.segmentation

    from scipy.io.wavfile import read
    from scipy import signal
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from scipy.io import wavfile
    from scipy.fftpack import fft
    import glob
    import librosa
    #from tkinter import tkk
    def handle_focus_in(_):
        BirdIDText.delete(0, tk.END)
        BirdIDText.config(fg='black')
    def handle_focus_out(_):
        if BirdIDText.get() == "":
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='grey')
            BirdIDText.insert(0, "Bird ID")
    def FileExplorerFunction():
        global song_folder
        song_folder = filedialog.askdirectory()
        FileDisplay = tk.Label(text=song_folder)
        FileDisplay.grid(row=2, column=1)
    def SegmentButtonClick():
        print("Button Pressed!")
        Bird_ID = BirdIDText.get()
        print(Bird_ID)
        global song_folder
        song_folder = song_folder + "/"
        print(song_folder)
        global segmenter
        segmenter = avn.segmentation.MFCCDerivative()
        global seg_data
        try:
            seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder,
                upper_threshold = MaxThresholdLabel.get(),
                lower_threshold = MinThresholdLabel.get())
        except:
            seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder,
                upper_threshold=0.1,
                lower_threshold=-0.1)
        # Default upper and lower thresholds are 0.1 and -0.1 respectively #
        out_file_dir = "C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/"
        print(seg_data.seg_table.head())
        try:
            seg_data.save_as_csv(out_file_dir)
        except:
            pass

    gui = tk.Tk()
    greeting = tk.Label(text="Welcome to the AVN gui!")
    greeting.grid(row=0)

    BirdIDText = tk.Entry(text="Bird ID", font=("Arial", 15), justify="center", fg="grey")
    BirdIDText.insert(0, "Bird ID")
    BirdIDText.bind("<FocusIn>", handle_focus_in)
    BirdIDText.bind("<FocusOut>", handle_focus_out)
    BirdIDText.grid(row=1, column=1)

    BirdIDLabel = tk.Label(text="Bird ID:")
    BirdIDLabel.grid(row=1, column=0)

    FileExplorer = tk.Button(text="Find Folder", command = lambda : FileExplorerFunction())
    FileExplorer.grid(row=2, column=0)

    SegmentButton = tk.Button(text="Segment!", command = lambda : SegmentButtonClick())
    SegmentButton.grid(row=5)

    MinThresholdText = tk.Entry(justify="center")
    MinThresholdText.grid(row=3, column=1)
    MinThresholdLabel = tk.Label(text="Min Threshold:")
    MinThresholdLabel.grid(row=3, column=0)

    MaxThresholdText = tk.Entry(justify="center")
    MaxThresholdText.grid(row=4, column=1)
    MaxThresholdLabel = tk.Label(text="Max Threshold:")
    MaxThresholdLabel.grid(row=4, column=0)

    def SpectrogramCreation():
        #print("Spectrogram")
        global song_folder
        #print(song_folder)
        filelist = []
        #directory = os.fsencode(song_folder)
        filelist = glob.glob(str(song_folder)+"/*.wav")
        newfilelist = []
        name1, name2 = filelist[1].split("\\")
        print("+++++++++++++++++++++++")
        FixedFileName = name1+"/"+name2
        for file in filelist:
            temp1, temp2 = file.split("\\")
            newfilelist.append(temp1+"/"+temp2)
        print(newfilelist[1:5])
        y, samplingrate = librosa.load(newfilelist[1])
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        image = librosa.display.specshow(D)
        global seg_data
        global segmenter
        #avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter, "MFCC Derivative")
        #spectrogram.savefig('fig.png')
        #image = tk.PhotoImage(file="fig.png")
        #imagelabel = ttk.Label(image=image)
        #imagelabel.grid()



        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        window_size = 1024
        window = np.hanning(window_size)
        stft = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='linear', x_axis='time')
        fig.savefig('spec.png')


    SpectrogramButton = tk.Button(text="Create Spectrogram", command = lambda : SpectrogramCreation())
    SpectrogramButton.grid(row=6)
    gui.mainloop()
except Exception:
    print(Exception)