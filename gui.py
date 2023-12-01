######  Required libraries to install: ########
# avn
# pymde
# hdbscan
# Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/


'''
For next time: Fix issue where labeling tab required input from segemntation tab file selection
    clean up code
    see if there's any way to optimize it
    figure out what parts of avn I updated so Therese can change avn accordingly
    make instructions for others to use gui
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
        print("Segmentation Complete!")
        print(seg_data.seg_table.head())
        try:
            seg_data.save_as_csv(out_file_dir)
            print("Successfully saved segmentation data!")
        except:
            print("Failed to save segmentation data!")
        shutil.rmtree(song_folder + "/TempSpectrogramFiles/")
        shutil.rmtree(str(song_folder) + 'TempFig.png')

    def SpectrogramDisplay(Direction):
        global ImgLabel
        global FileID
        global fig
        global Bird_ID
        global song_folder
        global newfilelist
        global seg_data
        global SpectrogramWindow
        try:
            ImgLabel.destroy()
        except:
            pass
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
        FolderSize = 20
        if FileID == 0:
            segFolderCount = 0
        if FileID % FolderSize == 0:
            segFolderCount = int(FileID // FolderSize)
        else:
            segFolderCount = int((FileID // FolderSize) + 1)

        try:
            os.makedirs(song_folder + "/TempSpectrogramFiles/")
        except: pass
        if segFolderCount != 0:
            for i in range(segFolderCount * FolderSize):
                segFolderCombined = song_folder + "/TempSpectrogramFiles/" + str(segFolderCount) + "/"
                try:
                    os.makedirs(segFolderCombined)
                except: pass
                shutil.copy(newfilelist[i], segFolderCombined)
        else:
            for i in range(FolderSize):
                segFolderCombined = song_folder + "/TempSpectrogramFiles/" + str(segFolderCount) + "/"
                try:
                    os.makedirs(segFolderCombined)
                except: pass
                shutil.copy(newfilelist[i], segFolderCombined)
        try:
            upper_threshold = float(MaxThresholdText.get())
        except:
            upper_threshold = 0.1
        try:
            lower_threshold = float(MinThresholdText.get())
        except:
            lower_threshold = -0.1
        segmenter = avn.segmentation.MFCCDerivative()
        Temp_seg_data = segmenter.make_segmentation_table(Bird_ID, segFolderCombined,
                                                     upper_threshold=upper_threshold,
                                                     lower_threshold=lower_threshold)
        # Default upper and lower thresholds are 0.1 and -0.1 respectively

        fig, ax, ax2, x_axis, spectrogram, axUpper, axLower, UT, LT = avn.segmentation.Plot.plot_seg_criteria(Temp_seg_data, segmenter,
                                                                                "MFCC Derivative",
                                                                                file_idx=FileID, figsize = (10,2.5),
                                                                                upper_threshold=upper_threshold, lower_threshold=lower_threshold)
        fig.savefig(str(song_folder)+"/"+'TempFig.png')
        print("Figure Saved:"+'TempFig.png')
        img = PhotoImage(file=str(song_folder)+"/"+'TempFig.png')
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
        #song_folder_path = song_folder #e.g. "C:/where_my_wav_files_are/Bird_ID/90/"
        tempsong_folder = segmentations_path.split("\\")
        segmentations_path = glob.glob(str(song_folder) + "/*" + "_seg_table.csv")
        try:
            segmentations_path = segmentations_path[0].replace("\\\\","/")
        except: pass
        song_folder = ""
        for f in tempsong_folder:
            song_folder = song_folder+f+"\\"
        output_file = song_folder+"/"+str(Bird_ID)+"_labels.csv" #e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"
        ######################################
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
        ###########################################
        def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
            spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
            spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

            # normalize
            S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

            return S_norm
        ############################################
        amin = 1e-5
        ref_db = 20
        min_level_db = -28
        win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
        hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
        n_fft = 512
        K = 10
        min_cluster_prop = 0.04
        embedding_dim = 2
        #############################################
        # load segmentations
        print(segmentations_path)
        predictions_reformat = pd.read_csv(segmentations_path)
        #predictions_reformat = predictions_reformat.rename(columns={"onset_s": 'onsets',
        #                                                            "offset_s": 'offsets',
        #                                                            "audio_path": 'files'})
        #predictions_reformat = predictions_reformat[predictions_reformat.label == 's']

        # Find any syllables with a duration less than 0.025s, and add on 0.01ms before the onset and 0.01ms after the offset
        predictions_reformat['onsets'][predictions_reformat['offsets'] - predictions_reformat['onsets'] < 0.025] = \
        predictions_reformat['onsets'] - 0.01

        segmentations = predictions_reformat

        # sometimes this padding can create negative onset times, which will cause errors.
        # correct this by setting any onsets <0 to 0.
        segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

        # Add syllable audio to dataframe
        syllable_dfs = pd.DataFrame()

        for song_file in segmentations.files.unique():
            file_path = song_folder + "/" + song_file
            song = avn.dataloading.SongFile(file_path)
            song.bandpass_filter(500, 15000)

            syllable_df = segmentations[segmentations['files'] == song_file]

            # this section is based on avgn.signalprocessing.create_spectrogram_dataset.get_row_audio()
            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                    for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
            syllable_dfs = pd.concat([syllable_dfs, syllable_df])

        # Normalize the audio  --- Ethan's comment: This won't work when there's an empty array for syllable_dfs_audio_values, so I'm just going to set those to '[0]'
        droppedRows = []
        #for i in range(len(syllable_dfs.audio.values)):
        #
        #    try:
        #        syllable_dfs['audio'][i] = librosa.util.normalize(syllable_dfs.audio.values[i])
        #    except:
        ##        print("Error! Audio values at line "+str(i)+" are invalid!"+" Line "+str(i)+" has been deleted.")
         #       syllable_dfs['audio'][i] = np.array([])
         #       droppedRows.append(i)
        syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
        # compute spectrogram for each syllable
        syllables_spec = []

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

        # flatten the spectrograms into 1D
        specs_flattened = [spec.flatten() for spec in syllables_spec_padded]
        specs_flattened_array = np.array(specs_flattened)

        # Embed
        mde = pymde.preserve_neighbors(specs_flattened_array, n_neighbors=K, embedding_dim=embedding_dim)
        embedding = mde.embed()

        # cluster
        min_cluster_size = math.floor(embedding.shape[0] * min_cluster_prop)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1).fit(embedding)
        print(embedding.shape)
        #for i in droppedRows:
        #    clusterer.pop(i)

        hdbscan_df = syllable_dfs

        # I had an issue where len(labels) was less than len(hdbscan_df), so I'm deleting any extra rows at the end to make them the same length
        #if len(hdbscan_df["files"]) != len(labels):
        #    if len(hdbscan_df["files"]) > len(labels):
        #        while a < len(hdbscan_df["files"]) - len(labels):
        #            hdbscan_df.drop(axis=0, index=len(hdbscan_df))
        #    if len(hdbscan_df["files"]) < len(labels):
        #        while a < len(labels) - len(hdbscan_df["files"]):
        #            hdbscan_df.drop(axis=0, index=len(hdbscan_df))
        hdbscan_df = syllable_dfs
        hdbscan_df["labels"] = clusterer.labels_

        hdbscan_df["X"] = embedding[:, 0]
        hdbscan_df["Y"] = embedding[:, 1]
        hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")

        hdbscan_df.to_csv(output_file)
        ################################################
        LabellingScatterplot = sns.scatterplot(data=hdbscan_df, x="X", y="Y", hue="labels", alpha=0.25, s=5)
        plt.title("My Bird's Syllables");
        LabellingFig = LabellingScatterplot.get_figure()
        LabellingFig.savefig("LabellingClusters.png")
        print("Figure Saved:" + 'LabellingClusters.png')
        LabellingImg = PhotoImage(file='LabellingClusters.png')
        LabellingImgLabel = tk.Label(LabelingFrame, image=LabellingImg)
        LabellingImgLabel.grid(columnspan=2)
        LabellingImgLabel.update()
        song_folder = song_folder[0:-1]
        try:
            os.makedirs(song_folder+"/LabellingPhotos")
        except: pass
        print(len(glob.glob(str(song_folder) + "/*.wav")))
        for i in range(len(glob.glob(str(song_folder) + "/*.wav"))):
            print(i)
            fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "", song_file_index=i, figsize=(20, 5), fontsize=14)
            fig.savefig(str(song_folder)+"/LabellingPhotos/"+str(song_file_name)+".png")
            print("Fig saved as "+str(song_file_name)+"!")

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