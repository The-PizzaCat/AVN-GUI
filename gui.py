######  Required libraries to install: ########
# avn
# pymde
# hdbscan
#  Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/


'''
For next time: Undo changes I made Monday and revert code to how it is from original labelling code
Remove unnecesary features like dropping rows that don't work
'''


'''
Things to change/update from Therese:
#DONE# 1. Generate spectrograms without needing to segment -- Do this last; super difficult
#DONE# 2. Display thresholds on spectrograms -- use axis data from fig object to add horizontal line at min/max thresholds
#DONE# 3. Change dimensions of spectrogram images by modifying hidden variable from avn segmentation
#DONE# 4. Add arrow buttons in spectrogram window that changes index of file selected for spectrogram instead of using tabs
	-- Update tab upon clicking arrow button
5. Have segmentation tables have unique names so that they don't get overwritten
	Name is based on Bird ID + Min Threshold + Max Threshold + Date



For Labeling Env:
1. Select Folder w/ song
2. Select Output segmentations
3. Run!

# DONE# ***Gui eventually should have a tab for segmentation and one for labeling***
***Add brief instructions for each tab (seg vs labeling)***
'''
import array

import numpy

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
    import pandas as pd
    pd.options.mode.chained_assignment = None
    import librosa
    import shutil
    import os
    import time

    def handle_focus_in(_):
        if BirdIDText.get() == "":
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
        FileDisplay = tk.Label(SegmentationFrame, text=song_folder)
        FileDisplay.grid(row=1, column=1)
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(song_folder) != None:
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')
            BirdIDText.insert(0, str(pattern.search(song_folder).group()))
            Bird_ID = str(pattern.search(song_folder).group())

    def LabelingFileExplorerFunction():
        global segmentations_path
        global Bird_ID
        segmentations_path = filedialog.askdirectory() #e.g. "C:/where_my_segmentation_table_is/Bird_ID_segmentations.csv"
        LabelingFileDisplay = tk.Label(LabelingFrame,text=segmentations_path)
        LabelingFileDisplay.grid(row=1, column=1)
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(segmentations_path) != None:
            LabelingBirdIDText.delete(0, tk.END)
            LabelingBirdIDText.config(fg='black')
            LabelingBirdIDText.insert(0, str(pattern.search(segmentations_path).group()))
            Bird_ID = str(pattern.search(segmentations_path).group())

    def SegmentButtonClick():
        print("Segmenting...")
        global Bird_ID
        Bird_ID = BirdIDText.get()
        global song_folder
        song_folder = song_folder + "/"
        global segmenter
        segmenter = avn.segmentation.MFCCDerivative()
        global seg_data
        try:
            MaxThreshold= float(MaxThresholdText.get())
        except:
            MaxThreshold=0.1
        try:
            MinThreshold= float(MinThresholdText.get())
        except:
            MinThreshold=-0.1
        seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder,
            upper_threshold=MaxThreshold,
            lower_threshold=MinThreshold)
        # Default upper and lower thresholds are 0.1 and -0.1 respectively #
        out_file_dir = song_folder
        #print(out_file_dir)
        print("Segmentation Complete!")
        print(seg_data.seg_table.head())
        try:
            seg_data.save_as_csv(out_file_dir)
            print("Successfully saved segmentation data!")
        except:
            print("Failed to save segmentation data!")

    def SpectrogramDisplay(Direction):
        global ImgLabel
        global FileID
        global fig
        global Bird_ID
        global song_folder
        global newfilelist
        global seg_data
        global SpectrogramWindow
        print(time.ctime(time.time()))
        try:
            ImgLabel.destroy()
        except:
            pass
        segmenter = avn.segmentation.MFCCDerivative()
        if Direction == "Left":
            FileID = FileID-1
        if Direction == "Right":
            FileID = FileID+1
        if Direction == "Start":
            FileID = 0
        try: # Checking if newfilelist exists, and if not will generate it with glob.glob()
            checkif_newfilelist_exists = newfilelist
        except:
            filelist = glob.glob(str(song_folder) + "/*.wav")
            newfilelist = []
            for file in filelist:
                temp1, temp2 = file.split("\\")
                newfilelist.append(temp1 + "/" + temp2)
        if FileID == 0:
            segFolder = 0
        if FileID%50 == 0:
            segFolder = int(FileID//50)
        else:
            segFolder = int((FileID//50) + 1)
        segFolderCombined = song_folder + "/TempSpectrogramFiles/"+str(segFolder)+"/"
        if segFolder != 0:
            for i in range(segFolder*50):
                shutil.copy(newfilelist[i], segFolderCombined)
        else:
            for i in range(50):
                shutil.copy(newfilelist[i], segFolderCombined)
        try:
            MaxThreshold= float(MaxThresholdText.get())
        except:
            MaxThreshold=0.1
        try:
            MinThreshold= float(MinThresholdText.get())
        except:
            MinThreshold=-0.1
        seg_data = segmenter.make_segmentation_table(Bird_ID, segFolderCombined,
                                                    upper_threshold=MaxThreshold,
                                                    lower_threshold=MinThreshold)
        # Default upper and lower thresholds are 0.1 and -0.1 respectively
        out_file_dir = song_folder+"/TempSpectrogramFiles"

        try:
            upper_threshold = float(MaxThresholdText.get())
        except:
            upper_threshold = 0.1
        try:
            lower_threshold = float(MinThresholdText.get())
        except:
            lower_threshold = -0.1
        fig, ax, ax2, x_axis, spectrogram, axUpper, axLower, UT, LT = avn.segmentation.Plot.plot_seg_criteria(seg_data, segmenter,
                                                                                "MFCC Derivative",
                                                                                file_idx=FileID, figsize = (10,2.5),
                                                                                upper_threshold=upper_threshold, lower_threshold=lower_threshold)
        #print(UT, LT)
        #print(upper_threshold, lower_threshold)
        #print(int(MaxThresholdText.get()), int(MinThresholdText.get()))
        fig.savefig('fig.png')
        print("Figure Saved:"+'fig.png')
        img = PhotoImage(file='fig.png')
        if FileID == 0:
            SpectrogramWindow = tk.Toplevel(gui)
            LeftButton = tk.Button(SpectrogramWindow, command=lambda: SpectrogramDisplay("Left"), text="Previous")
            LeftButton.grid(row=0, column=50)
            RightButton = tk.Button(SpectrogramWindow, command=lambda: SpectrogramDisplay("Right"), text="Next")
            RightButton.grid(row=0, column=51)
        ImgLabel = tk.Label(SpectrogramWindow, image=img)
        ImgLabel.grid(row=1, columnspan=100)
        ImgLabel.update()

        SpectrogramWindow.mainloop()


### All code for labeling comes from Single_Directory_Template_for_AVN_labeling file in locally installed avn files
    # located at C:\Users\ethan\Desktop\Roberts_Lab_Files\AVNGUI\Local_AVN_Installation_Materials
    def labelingPlaceholder():
        global segmentations_path
        global Bird_ID
        global song_folder
        global LabelingFileDisplay
        print(segmentations_path)
        #song_folder_path = song_folder #e.g. "C:/where_my_wav_files_are/Bird_ID/90/"
        tempsong_folder = segmentations_path.split("\\")

        ### Issue with the line below: I now name seg tables with thresholds in the name, so names differ -- need to use regular expressions to ensure glob.glob can always find file
        segmentations_path = glob.glob(str(song_folder) + "/*" + "_seg_table.csv")
        print(segmentations_path)
        try:
            segmentations_path = segmentations_path[0].replace("\\\\","/")
        except: pass
        song_folder = ""
        for f in tempsong_folder:
            song_folder = song_folder+f+"\\"
        output_file = song_folder+"/"+str(Bird_ID)+"_labels.csv" #e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"

        import pymde
        import numpy as np
        import pandas as pd
        import librosa
        import matplotlib
        import matplotlib.pyplot as plt
        import avn.dataloading
        import avn.plotting
        import hdbscan
        import math
        import sklearn
        import seaborn as sns


        def make_spec(syll_wav, hop_length = 128, win_length = 512, n_fft = 512, amin = 1e-5, ref_db = 20, min_level_db = -28):
            spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
            spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

            # normalize
            S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

            return S_norm

        amin = 1e-5
        ref_db = 20
        min_level_db = -28
        win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
        hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
        n_fft = 512
        K = 10
        min_cluster_prop = 0.04
        embedding_dim = 2

        # load segmentations
        print(segmentations_path)
        predictions_reformat = pd.read_csv(segmentations_path)
        predictions_reformat = predictions_reformat.rename(columns={"onset_s": 'onsets',
                                                                    "offset_s": 'offsets',
                                                                    "audio_path": 'files'})
        # This line does not work; also not sure what it does...
        #predictions_reformat = predictions_reformat[predictions_reformat.label == 's']

        # Find any syllables with a duration less than 0.025s, and add on 0.01ms before the onset and 0.01ms after the offset
        predictions_reformat['onsets'][predictions_reformat['offsets'] - predictions_reformat['onsets'] < 0.025] = \
        predictions_reformat['onsets'] - 0.01
        predictions_reformat['offsets'][predictions_reformat['offsets'] - predictions_reformat['onsets'] < 0.025] = \
        predictions_reformat['offsets'] + 0.01

        segmentations = predictions_reformat

        # sometimes this padding can create negative onset times, which will cause errors.
        # correct this by setting any onsets <0 to 0.
        segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

        # Add syllable audio to dataframe
        syllable_dfs = pd.DataFrame()
        ###################
        for song_file in segmentations.files.unique():
            file_path = segmentations_path.split("\\")[0] + "/" + song_file
            song = avn.dataloading.SongFile(file_path)
            song.bandpass_filter(500, 15000)

            syllable_df = segmentations[segmentations['files'] == song_file]

            # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                    for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
            syllable_dfs = pd.concat([syllable_dfs, syllable_df])
        print(syllable_dfs)
        print("---------")


        # Normalize the audio  --- Ethan's comment: This won't work when there's an empty array for syllable_dfs_audio_values, so I'm just going to ignore those
        droppedRows = []
        for i in range(len(syllable_dfs.audio.values)):

            try:
                syllable_dfs['audio'][i] = librosa.util.normalize(syllable_dfs.audio.values[i])
            except:
                print("Error! Audio values at line "+str(i)+" are invalid!")
                syllable_dfs['audio'][i] = [0]
                print("Values at line "+str(i)+" have been set to "+str('[0]'))
        ##syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]

        # compute spectrogram for each syllable
        syllables_spec = []
        #print(syllable_dfs.audio.values[1:5])
        #print(type(syllable_dfs.audio.values[1:5]))
        for syllable in syllable_dfs.audio.values:
            if len(syllable) > 0:
                syllable_spec = make_spec(syllable,
                                      hop_length=hop_length,
                                      win_length=win_length,
                                      n_fft=n_fft,
                                      ref_db=ref_db,
                                      amin=amin,
                                      min_level_db=min_level_db)
                if syllable_spec.shape[1] > 300:
                    print("Long Syllable Corrections! Spectrogram Duration = " + str(syllable_spec.shape[1]))
                    syllable_spec = syllable_spec[:, :300]

                syllables_spec.append(syllable_spec)


        # normalize spectrograms
        def norm(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))


        syllables_spec_norm = [norm(i) for i in syllables_spec]

        # Pad spectrograms for uniform dimensions
        spec_lens = [np.shape(i)[1] for i in syllables_spec]
        pad_length = np.max(spec_lens)

        syllables_spec_padded = []

        for spec in syllables_spec_norm:
            to_add = pad_length - np.shape(spec)[1]
            pad_left = np.floor(float(to_add) / 2).astype("int")
            pad_right = np.ceil(float(to_add) / 2).astype("int")
            spec_padded = np.pad(spec, [(0, 0), (pad_left, pad_right)], 'constant', constant_values=0)
            syllables_spec_padded.append(spec_padded)
        print("354")
        print(time.ctime(time.time()))
        # flatten the spectrograms into 1D
        specs_flattened = [spec.flatten() for spec in syllables_spec_padded]
        specs_flattened_array = np.array(specs_flattened)
        print("358")
        print(time.ctime(time.time()))
        # Embed
        mde = pymde.preserve_neighbors(specs_flattened_array, n_neighbors=K, embedding_dim=embedding_dim)
        embedding = mde.embed()
        print("362")
        print(time.ctime(time.time()))
        # cluster
        min_cluster_size = math.floor(embedding.shape[0] * min_cluster_prop)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1).fit(embedding)
        print("366")
        print(time.ctime(time.time()))
        #clusterer.drop(droppedRows)


        # Delete this # Instead, have a warning/error print
        #for i in droppedRows:
        #    syllable_dfs = np.delete(syllable_dfs, i, 0)
        #    print(i)
        #hdbscan_df = pd.DataFrame(syllable_dfs, columns =['Start','Stop','File','Segments','Temp'])
        hdbscan_df = syllable_dfs

        labels = clusterer.labels_
        #print(type(hdbscan_df))

        # I had an issue where len(labels) was less than len(hdbscan_df), so I'm deleting any extra rows at the end to make them the same length
        if len(hdbscan_df["files"]) != len(labels):
            if len(hdbscan_df["files"]) > len(labels):
                while a < len(hdbscan_df["files"]) - len(labels):
                    hdbscan_df.drop(axis=0, index=len(hdbscan_df))
            if len(hdbscan_df["files"]) < len(labels):
                while a < len(labels) - len(hdbscan_df["files"]):
                    hdbscan_df.drop(axis=0, index=len(hdbscan_df))
        print(len(hdbscan_df["files"]))
        print(len(labels))
        hdbscan_df[["labels"]] = labels

        print('369')
        print(time.ctime(time.time()))
        #X_col = embedding[:, 0]
        #Y_col = embedding[:, 1]
        #hdbscan_df = np.column_stack((hdbscan_df, X_col)) #Column #7
        #hdbscan_df = np.column_stack((hdbscan_df, Y_col)) #Column #8
        hdbscan_df["X"] = embedding[:, 0]
        hdbscan_df["Y"] = embedding[:, 1]
        hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")
        print('373')
        print(time.ctime(time.time()))
        print(type(hdbscan_df))
        #try:
        print(type(hdbscan_df))
        hdbscan_df.to_csv(output_file)
        #except:
        #    numpy.savetxt(str(output_file), hdbscan_df, fmt="%s")
        print('375')
        print(time.ctime(time.time()))

        LabellingScatterplot = sns.scatterplot(data=hdbscan_df, x=hdbscan_df[:,7], y=hdbscan_df[:,8], hue=hdbscan_df[:,6], alpha=0.25, s=5)
        plt.title("My Bird's Syllables");
        LabellingFig = LabellingScatterplot.get_figure()
        LabellingFig.savefig("Labelling.png")
        print("Figure Saved:" + 'Labelling.png')
        LabellingImg = PhotoImage(file='Labelling.png')
        LabellingImgLabel = tk.Label(LabelingFrame, image=LabellingImg)
        LabellingImgLabel.grid()
        LabellingImgLabel.update()

        print(hdbscan_df[1:5, 1])
        print(hdbscan_df[1:5, 2])
        print(hdbscan_df[1:5, 3])
        print(hdbscan_df[1:5, 4])
        print(hdbscan_df[1:5, 5])
        print(hdbscan_df[1:5, 6])

        print("378")
        print(time.ctime(time.time()))
        # "Files" column is 3rd column in hdbscan_df
        for i in range(3):
            avn.plotting.plot_spectrogram_with_labels(hdbscan_df, segmentations_path, Bird_ID, song_file_index=i, figsize=(20, 5),
                                                      fontsize=14)



    gui = tk.Tk()
    notebook = ttk.Notebook(gui)
    notebook.grid()
    SegmentationFrame = tk.Frame(gui)
    notebook.add(SegmentationFrame, text="Segmentation")
    gui.title("AVN Segmentation and Labeling")
    greeting = tk.Label(SegmentationFrame, text="Welcome to the AVN gui!")
    greeting.grid(row=0, columnspan=2)

    BirdIDText = tk.Entry(SegmentationFrame, font=("Arial", 15), justify="center", fg="grey")
    BirdIDText.insert(0, "Bird ID")
    BirdIDText.bind("<FocusIn>", handle_focus_in)
    BirdIDText.bind("<FocusOut>", handle_focus_out)
    BirdIDText.grid(row=2, column=1)

    BirdIDLabel = tk.Label(SegmentationFrame, text="Bird ID:")
    BirdIDLabel.grid(row=2, column=0)

    FileExplorer = tk.Button(SegmentationFrame, text="Find Folder", command = lambda : FileExplorerFunction())
    FileExplorer.grid(row=1, column=0)

    SegmentButton = tk.Button(SegmentationFrame, text="Segment!", command = lambda : SegmentButtonClick())
    SegmentButton.grid(row=6, columnspan=2)

    MinThresholdText = tk.Entry(SegmentationFrame, justify="center", fg='grey', font=("Arial", 15))
    MinThresholdText.insert(0, "-0.1")
    MinThresholdText.grid(row=3, column=1)
    MinThresholdLabel = tk.Label(SegmentationFrame, text="Min Threshold:")
    MinThresholdText.bind("<FocusIn>", MinFocusIn)
    MinThresholdText.bind("<FocusOut>", MinFocusOut)
    MinThresholdLabel.grid(row=3, column=0)

    MaxThresholdText = tk.Entry(SegmentationFrame, justify="center", fg='grey', font=("Arial", 15))
    MaxThresholdText.insert(0, "0.1")
    MaxThresholdText.grid(row=4, column=1)
    MaxThresholdLabel = tk.Label(SegmentationFrame, text="Max Threshold:")
    MaxThresholdText.bind("<FocusIn>", MaxFocusIn)
    MaxThresholdText.bind("<FocusOut>", MaxFocusOut)
    MaxThresholdLabel.grid(row=4, column=0)

    SpectrogramButton = tk.Button(SegmentationFrame, text="Create Spectrogram", command = lambda : SpectrogramDisplay("Start"))
    SpectrogramButton.grid(row=5, columnspan=2)

    ### Labeling Window###
    LabelingFrame = tk.Frame()
    notebook.add(LabelingFrame, text="Labeling")
    LabelingButton = tk.Button(LabelingFrame, text="Labeling", command = lambda : labelingPlaceholder())
    LabelingButton.grid(row=3, column = 0,columnspan=2)
    Labelinggreeting = tk.Label(LabelingFrame, text="Welcome to the AVN gui!")
    Labelinggreeting.grid(row=0, column=0, columnspan=2)

    LabelingBirdIDText = tk.Entry(LabelingFrame, font=("Arial", 15), justify="center", fg="grey")
    LabelingBirdIDText.insert(0, "Bird ID")
    LabelingBirdIDText.bind("<FocusIn>", handle_focus_in)
    LabelingBirdIDText.bind("<FocusOut>", handle_focus_out)
    LabelingBirdIDText.grid(row=2, column=1)

    LabelingBirdIDLabel = tk.Label(LabelingFrame, text="Bird ID:")
    LabelingBirdIDLabel.grid(row=2, column=0)

    LabelingFileExplorer = tk.Button(LabelingFrame, text="Find Folder", command=lambda: LabelingFileExplorerFunction())
    LabelingFileExplorer.grid(row=1, column=0)



    #def on_closing():
    #    files = glob.glob(song_folder + "/TempSpectrogramFiles")
    #    for f in files:
    #        os.remove(f)
    #    gui.destroy()
    #gui.protocol("WM_DELETE_WINDOW", on_closing)

    gui.mainloop()


except Exception:
    print(Exception)