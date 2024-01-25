######  Required libraries to install: ########
# avn
# pymde
# hdbscan
# Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Fixes error about no torch module found: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

'''
Updates to Add:
 - #DONE# Feature to print all spectrograms in a folder without any overlay (plain)
 - #DONE# Preview for labeling module similar to that of segmentation, where people can preview labeled spectrograms
 - Loading bar?
 - Modify function to find Bird ID within file directory so that it finds ID from .wav file names rather than folder name
 - #DONE# Create tab groups that contain sub-tabs (e.g. info tab and advanced settings tab for each module, combine acoustic features of single vs multi file, etc.)
    - Consider simply having tabs show/hide when you click on/off a parent tab
 - Make spectrogram (and eventually labeling when I've added it) display images full-resolution
    - Because images are so large, only preview first ~5 seconds or so
 - #DONE# GPU acceleration (especially helpful for when WhisperSeg is implemented)
 - #DONE# Transition to WhisperSeg
 - Display filename above segmentation/labeling display
 - #DONE# Export as .exe
 - Add message when photos are saved from segmentation or labeling display
'''

try:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import *
    from tkinter import ttk
    import csv
    import numpy as np
    import glob
    import re
    import pandas as pd
    pd.options.mode.chained_assignment = None
    import librosa
    # import shutil
    import os
    import time
    import datetime

    # import customtkinter
    # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # import pymde
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import hdbscan
    # import math
    # import sklearn
    # import seaborn as sns
    # from matplotlib.figure import Figure

    def handle_focus_in(_):
        if BirdIDText.get() == "Bird ID":
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')

    def handle_focus_out(_):
        if BirdIDText.get() == "":
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='grey')
            BirdIDText.insert(0, "Bird ID")

    def FileExplorerFunction():
        global song_folder
        global Bird_ID
        song_folder = filedialog.askdirectory()
        FileDisplay = tk.Label(SegmentationMainFrame, text=song_folder)
        FileDisplay.grid(row=1, column=1, sticky="w")

        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(song_folder) != None:
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')
            BirdIDText.insert(0, str(pattern.search(song_folder).group()))
            Bird_ID = str(pattern.search(song_folder).group())

    def SegmentButtonClick():  ### This function has been modified to use Whisper via cpu
        SegmentingProgressLabel = tk.Label(SegmentationFrame, text="Segmenting...", font=("Arial", 10), justify=CENTER)
        SegmentingProgressLabel.grid(row=7, column=0, columnspan=2)

        global Bird_ID
        Bird_ID = BirdIDText.get()
        global song_folder
        song_folder = song_folder + "/"

        # For Whisper:
        sr = 32000
        min_frequency = 0
        spec_time_step = 0.0025
        min_segment_length = 0.01
        eps = 0.02
        num_trials = 3

        import sys
        cwd = os.getcwd()
        WhisperDir = os.path.join(cwd, "WhisperSeg-master")
        sys.path.insert(0, str(WhisperDir))

        from model import WhisperSegmenterFast
        global SegmentationStyle_Var
        if SegmentationStyle_Var.get() == 0: # Segmentation style is CPU
            segmenter = WhisperSegmenterFast("nccratliri/whisperseg-large-ms-ct2", device="cpu")
        elif SegmentationStyle_Var.get() == 1: # Segmentation style is GPU
            segmenter = WhisperSegmenterFast("nccratliri/whisperseg-large-ms-ct2", device="cuda")

        # get list of files to segment
        song_folder_path = song_folder

        # Create output folder -- this will be used by all modules later on
        try: # Makes output folder if it doesn't exist already
            OutputPath = os.path.join(song_folder_path, "GUI_Output")
            os.mkdir(OutputPath)
        except: pass # if output folder already exists, the above code raises an error and is ignored

        # Create folder for segmentation output within parent output folder
        try:
            OutputPath = os.path.join(song_folder_path+"GUI_Output/", "Segmentations")
            os.mkdir(OutputPath)
        except: pass

        global SplitSegmentations
        print(SplitSegmentations.get())
        if SplitSegmentations.get() == 0:
            all_songs = glob.glob(song_folder_path + "/**/*.wav", recursive=True)

            # initialize empty dataframe to store segments
            full_seg_table = pd.DataFrame()

            # loop over each file in folder
            Progress_Var = IntVar()
            Progress_Var.set(0)
            ProgressBar = ttk.Progressbar(SegmentationFrame, variable=Progress_Var, maximum=len(all_songs))
            ProgressBar.grid()
            time.sleep(0.1)
            for i, song in enumerate(all_songs):
                # load audio
                audio, __ = librosa.load(song, sr=sr)
                # segment file
                prediction = segmenter.segment(audio, sr=sr, min_frequency=min_frequency, spec_time_step=spec_time_step,
                                               min_segment_length=min_segment_length, eps=eps, num_trials=num_trials)
                # format segmentation as dataframe
                curr_prediction_df = pd.DataFrame(prediction)
                # add file name to dataframe
                song_name = song.split("\\")[-1]
                curr_prediction_df['file'] = song_name
                # add full file directory to dataframe ### Added by Ethan ###
                curr_prediction_df['directory'] = song[:-(len(song_name))]
                # add current file's segments to full_seg_table
                full_seg_table = pd.concat([full_seg_table, curr_prediction_df])
                Progress_Var.set(i)
                ProgressBar.update()
                time.sleep(0.1)

            SegmentingProgressLabel.config(text="Segmentation Complete!")
            ProgressBar.destroy()
            time.sleep(1)
            # save full_seg_table
            try:
                full_seg_table.to_csv(song_folder_path+"GUI_Output/Segmentations/" + str(Bird_ID) + "_wseg" + "_" + str(datetime.datetime.now().year)
                + "_" + str(datetime.datetime.now().month)+ "_" + str(datetime.datetime.now().day)+ "___" + str(datetime.datetime.now().hour)+
                                      "_" + str(datetime.datetime.now().minute)+ "_" + str(datetime.datetime.now().second)+".csv")
                SegmentingProgressLabel.config(text="Successfully Saved Segmentation Data!")
                full_seg_table = None
                prediction = None
            except:
                SegmentingProgressLabel.config(text="Failed to Save Segmentation Data")

            SegmentingProgressLabel.update()
        elif SplitSegmentations.get() == 1:
            all_songs = glob.glob(song_folder_path + "/**/*.wav", recursive=True)

            # Gathers all subfolders that contain .wav files
            song_directories = []
            for song in all_songs:
                song = song.replace("\\","/")
                song = song.split("/")[:-1]
                song_merged = ""
                for x in song:
                    song_merged = song_merged+x+"/"
                if song_merged not in song_directories:
                    song_directories.append(song_merged)

            print(song_directories)
            # Runs segmentations for each subfolder
            for directory in song_directories:
                print(directory)
                # initialize empty dataframe to store segments
                full_seg_table = pd.DataFrame()

                # loop over each file in folder
                try:
                    Progress_Var.set(0)
                except:
                    Progress_Var = IntVar()
                    Progress_Var.set(0)
                    ProgressBar = ttk.Progressbar(SegmentationFrame, variable=Progress_Var, maximum=len(all_songs))
                    ProgressBar.grid()
                try:
                    ProgressBarLabel.destroy()
                except: pass
                ProgressBarLabel = tk.Label(SegmentationFrame, text="Processing Folder "+str(song_directories.index(directory)+1)+ " of "+str(len(song_directories)))
                ProgressBarLabel.grid()
                time.sleep(0.1)

                all_songs_sub = glob.glob(directory + "/**.wav")
                for i, song in enumerate(all_songs_sub):
                    # load audio
                    audio, __ = librosa.load(song, sr=sr)
                    # segment file
                    prediction = segmenter.segment(audio, sr=sr, min_frequency=min_frequency, spec_time_step=spec_time_step,
                                                   min_segment_length=min_segment_length, eps=eps, num_trials=num_trials)
                    # format segmentation as dataframe
                    curr_prediction_df = pd.DataFrame(prediction)
                    # add file name to dataframe
                    song_name = song.split("\\")[-1]
                    curr_prediction_df['file'] = song_name
                    # add full file directory to dataframe ### Added by Ethan ###
                    curr_prediction_df['directory'] = song[:-(len(song_name))]
                    # add current file's segments to full_seg_table
                    full_seg_table = pd.concat([full_seg_table, curr_prediction_df])
                    Progress_Var.set(i)
                    ProgressBar.update()
                    time.sleep(0.1)

                # save full_seg_table
                try:
                    directory_corrected = ""
                    for a in directory.split("/"):
                        directory_corrected = directory_corrected + a + "-"
                    directory_corrected.replace(":","")
                    print(directory_corrected)
                    print(song_folder_path)
                    print("+-+-+-+-+-")
                    # print(full_seg_table)
                    # print(song_folder_path+"GUI_Output/Segmentations/" + str(Bird_ID) + "_seg_" + str(directory_corrected) + "_" + str(datetime.datetime.now().year)
                                    #+ "_" + str(datetime.datetime.now().month)+ "_" + str(datetime.datetime.now().day)+ "___" + str(datetime.datetime.now().hour)+
                                    #  "_" + str(datetime.datetime.now().minute)+ "_" + str(datetime.datetime.now().second)+"_wseg.csv")
                    full_seg_table.to_csv(song_folder_path+"GUI_Output/Segmentations/" + str(Bird_ID) + "_seg_" + str(directory.split("/")[-1])+"_wseg.csv")
                    print("-/-/-/-/-/-/-/-/")
                    SegmentingProgressLabel.config(text="Successfully Saved Segmentation Data!")
                    full_seg_table = None
                    prediction = None
                except:
                    print("Failed to Save Segmentation Data")
                    SegmentingProgressLabel.config(text="Failed to Save Segmentation Data")
                    SegmentingProgressLabel.update()

                SegmentingProgressLabel.update()

            SegmentingProgressLabel.config(text="Segmentation Complete!")
            ProgressBar.destroy()
            time.sleep(1)
        else:
            print("Error!")

    ### Initialize gui ###
    gui = tk.Tk()
    gui.title("AVN Segmentation and Labeling")

    ParentStyle = ttk.Style()
    ParentStyle.configure('Parent.TNotebook.Tab', font=('Arial', '10', 'bold'))

    notebook = ttk.Notebook(gui, style='Parent.TNotebook.Tab')
    notebook.grid()

    MasterFrameWidth = 600
    MasterFrameHeight = 300

    ### Segmentation Window ###
    SegmentationFrame = tk.Frame(gui)
    notebook.add(SegmentationFrame, text="Segmentation")

    SegmentationMainFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook = ttk.Notebook(SegmentationFrame)
    SegmentationNotebook.grid(row=1)
    SegmentationNotebook.add(SegmentationMainFrame, text="Home")
    SegmentationMainFrame.grid_propagate(False)
    SegmentationSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook.add(SegmentationSettingsFrame, text="Advanced Settings")
    SegmentationInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook.add(SegmentationInfoFrame, text="Info")

    BirdIDText = tk.Entry(SegmentationMainFrame, font=("Arial", 15), justify="center", fg="grey")
    BirdIDText.insert(0, "Bird ID")
    BirdIDText.bind("<FocusIn>", handle_focus_in)
    BirdIDText.bind("<FocusOut>", handle_focus_out)
    BirdIDText.grid(row=2, column=1, sticky="w", columnspan=2)

    BirdIDLabel = tk.Label(SegmentationMainFrame, text="Bird ID:")
    BirdIDLabel.grid(row=2, column=0)

    FileExplorer = tk.Button(SegmentationMainFrame, text="Find Folder", command=lambda: FileExplorerFunction())
    FileExplorer.grid(row=1, column=0)

    global SplitSegmentations
    SplitSegmentations = IntVar()
    Split_Segmentations = tk.Checkbutton(SegmentationMainFrame, text="Split Segments", variable=SplitSegmentations, onvalue=1, offvalue=0)
    Split_Segmentations.grid(row=6, column=1)

    SegmentationStyle_Label = tk.Label(SegmentationMainFrame, text="Segmentation Style:")
    SegmentationStyle_Label.grid(row=7, column=0)
    global SegmentationStyle_Var
    SegmentationStyle_Var = tk.IntVar()
    SegmentationStyle_CPU = tk.Radiobutton(SegmentationMainFrame, text="CPU", variable=SegmentationStyle_Var, value=0)
    SegmentationStyle_CPU.grid(row=7, column=1)
    SegmentationStyle_GPU = tk.Radiobutton(SegmentationMainFrame, text="GPU", variable=SegmentationStyle_Var, value=1)
    SegmentationStyle_GPU.grid(row=7, column=2)

    SegmentButton = tk.Button(SegmentationMainFrame, text="Segment!", command=lambda: SegmentButtonClick())
    SegmentButton.grid(row=8, column=1)

    ttk.Style().theme_use("clam")
    gui.mainloop()

except Exception:
    print(Exception)