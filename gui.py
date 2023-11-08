try:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import *
    from tkinter import ttk
    import avn
    import avn.dataloading
    import avn.segmentation
    import csv
    import numpy as np
    import glob
    import re

    def handle_focus_in(_):
        BirdIDText.delete(0, tk.END)
        BirdIDText.config(fg='black')
    def handle_focus_out(_):
        if BirdIDText.get() == "":
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='grey')
            BirdIDText.insert(0, "Bird ID")
    def MinFocusIn(_):
        MinThresholdText.delete(0, tk.END)
        MinThresholdText.config(fg='black')
    def MinFocusOut(_):
        if MinThresholdText.get() == "":
            MinThresholdText.config(fg='grey')
            MinThresholdText.insert(0, "-0.1")
    def MaxFocusIn(_):
        MaxThresholdText.delete(0, tk.END)
        MaxThresholdText.config(fg='black')
    def MaxFocusOut(_):
        if MaxThresholdText.get() == "":
            MaxThresholdText.config(fg='grey')
            MaxThresholdText.insert(0, "0.1")

    def FileExplorerFunction():
        global song_folder
        global Bird_ID
        song_folder = filedialog.askdirectory()
        FileDisplay = tk.Label(text=song_folder)
        FileDisplay.grid(row=1, column=1)
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(song_folder) != None:
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')
            BirdIDText.insert(0, str(pattern.search(song_folder).group()))
            Bird_ID = str(pattern.search(song_folder).group())

    def SegmentButtonClick():
        print("Segmenting...")
        global Bird_ID
        Bird_ID = BirdIDText.get()
        print(Bird_ID)
        global song_folder
        song_folder = song_folder + "/"
        #print(song_folder)
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
        out_file_dir = song_folder
        print("Segmentation Complete!")
        print(seg_data.seg_table.head())
        try:
            seg_data.save_as_csv(out_file_dir)
        except:
            pass

    gui = tk.Tk()
    gui.title("AVN Segmentation")
    greeting = tk.Label(text="Welcome to the AVN gui!")
    greeting.grid(row=0)

    BirdIDText = tk.Entry(text="Bird ID", font=("Arial", 15), justify="center", fg="grey")
    BirdIDText.insert(0, "Bird ID")
    BirdIDText.bind("<FocusIn>", handle_focus_in)
    BirdIDText.bind("<FocusOut>", handle_focus_out)
    BirdIDText.grid(row=2, column=1)

    BirdIDLabel = tk.Label(text="Bird ID:")
    BirdIDLabel.grid(row=2, column=0)

    FileExplorer = tk.Button(text="Find Folder", command = lambda : FileExplorerFunction())
    FileExplorer.grid(row=1, column=0)

    SegmentButton = tk.Button(text="Segment!", command = lambda : SegmentButtonClick())
    SegmentButton.grid(row=5)

    MinThresholdText = tk.Entry(justify="center", fg='grey', text='-0.1', font=("Arial", 15))
    MinThresholdText.insert(0, "-0.1")
    MinThresholdText.grid(row=3, column=1)
    MinThresholdLabel = tk.Label(text="Min Threshold:")
    MinThresholdText.bind("<FocusIn>", MinFocusIn)
    MinThresholdText.bind("<FocusOut>", MinFocusOut)
    MinThresholdLabel.grid(row=3, column=0)

    MaxThresholdText = tk.Entry(justify="center", fg='grey', text='0.1', font=("Arial", 15))
    MaxThresholdText.insert(0, "0.1")
    MaxThresholdText.grid(row=4, column=1)
    MaxThresholdLabel = tk.Label(text="Max Threshold:")
    MaxThresholdText.bind("<FocusIn>", MaxFocusIn)
    MaxThresholdText.bind("<FocusOut>", MaxFocusOut)
    MaxThresholdLabel.grid(row=4, column=0)

    def SpectrogramCreation():
        global song_folder
        filelist = glob.glob(str(song_folder)+"/*.wav")
        newfilelist = []
        print("+++++++++++++++++++++++")
        for file in filelist:
            temp1, temp2 = file.split("\\")
            newfilelist.append(temp1+"/"+temp2)
        print(newfilelist[1:5])
        global seg_data
        global segmenter

        #AVN files are located here: C:\Users\ethan\anaconda3\envs\UseThisForGuiTesting\Lib\site-packages\avn'''

        SpectrogramWindow = tk.Toplevel(gui)
        tabControl = ttk.Notebook(SpectrogramWindow)
        TabNames = []
        PlotLabelNames = []
        '''for i in range(1,4):
            print(i)
            fig, ax, ax2, x_axis, spectrogram = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                        "MFCC Derivative", file_idx = (i-1))
            fig.savefig('fig'+str(i)+'.png')
            print("figure saved")
            TabNames.append("tab"+str(i))
            temptab = str(TabNames[i-1])
            temptab = ttk.Frame(tabControl)
            tabControl.add(temptab, text='Tab '+str(i))
            img = PhotoImage(file='fig'+str(i)+'.png')
            smaller_img = img.subsample(2, 2)
            PlotLabelNames.append("PlotLabel"+str(i))
            TempPlotLabelName = str(PlotLabelNames[i-1])
            TempPlotLabelName = tk.Label(temptab, image=smaller_img)
            #tabControl.pack(expand=1, fill="both")
            TempPlotLabelName.pack()
            SpectrogramWindow.update()
        tabControl.pack(expand=1, fill="both")
'''
        tab1 = ttk.Frame(tabControl)
        tabControl.add(tab1, text="tab1")
        tab2 = ttk.Frame(tabControl)
        tabControl.add(tab2, text="tab2")
        tab3 = ttk.Frame(tabControl)
        tabControl.add(tab3, text="tab3")
        tab4 = ttk.Frame(tabControl)
        tabControl.add(tab4, text="tab4")
        tab5 = ttk.Frame(tabControl)
        tabControl.add(tab5, text="tab5")

        fig, ax, ax2, x_axis, spectrogram = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                        "MFCC Derivative",
                                                                                        file_idx=0)
        fig.savefig('fig0.png')
        img0 = PhotoImage(file='fig0.png')
        smaller_img0 = img0.subsample(2, 2)
        label0 = tk.Label(tab1, image=smaller_img0)
        label0.pack()
        #############
        fig, ax, ax2, x_axis, spectrogram = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                    "MFCC Derivative",
                                                                                    file_idx=1)
        fig.savefig('fig1.png')
        img1 = PhotoImage(file='fig1.png')
        smaller_img1 = img1.subsample(2, 2)
        label1 = tk.Label(tab2, image=smaller_img1)
        label1.pack()
        #############
        fig, ax, ax2, x_axis, spectrogram = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                    "MFCC Derivative",
                                                                                    file_idx=2)
        fig.savefig('fig2.png')
        img2 = PhotoImage(file='fig2.png')
        smaller_img2 = img2.subsample(2, 2)
        label2 = tk.Label(tab3, image=smaller_img2)
        label2.pack()
        #############
        fig, ax, ax2, x_axis, spectrogram = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                    "MFCC Derivative",
                                                                                    file_idx=3)
        fig.savefig('fig3.png')
        img3 = PhotoImage(file='fig3.png')
        smaller_img3 = img3.subsample(2, 2)
        label3 = tk.Label(tab4, image=smaller_img3)
        label3.pack()
        #############

        #### Add argument in avn.segmentation.Plot.plot_seg_criteria to change image dimensions

        tabControl.pack(expand=1, fill="both")


        SpectrogramWindow.mainloop()
    SpectrogramButton = tk.Button(text="Create Spectrogram", command = lambda : SpectrogramCreation())
    SpectrogramButton.grid(row=6)


    gui.mainloop()
except Exception:
    print(Exception)