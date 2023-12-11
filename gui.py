######  Required libraries to install: ########
# avn
# pymde
# hdbscan
# Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/

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
 - GPU acceleration (especially helpful for when WhisperSeg is implemented)
 - Transition to WhisperSeg
 - Display filename above segmentation/labeling display
 - #DONE# Export as .exe
 - Add message when photos are saved from segmentation or labeling display
'''



try:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import *
    from tkinter import ttk
    import avn
    import avn.segmentation
    import avn.dataloading as dataloading
    import avn.acoustics as acoustics
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
    import customtkinter
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import pymde
    import matplotlib
    import matplotlib.pyplot as plt
    import avn.dataloading as dataloading
    import avn.plotting
    import hdbscan
    import math
    import sklearn
    import seaborn as sns


    def handle_focus_in(_):
        if BirdIDText.get() == "Bird ID":
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

    def LabelingCountFocusIn(_):
        global tempLabelingPhotoCount
        tempLabelingPhotoCount = LabelingPhotoCount.get()
        if tempLabelingPhotoCount == "0":
            LabelingPhotoCount.delete(0, tk.END)

    def LabelingCountFocusOut(_):
        global tempLabelingPhotoCount
        if tempLabelingPhotoCount == "0" and LabelingPhotoCount.get() == "":
            LabelingPhotoCount.insert(0,"0")

    def labeling_handle_focus_in(_):
        if LabelingBirdIDText.get() == "Bird ID":
            LabelingBirdIDText.delete(0, tk.END)
            LabelingBirdIDText.config(fg='black')

    def labeling_handle_focus_out(_):
        if LabelingBirdIDText.get() == "":
            LabelingBirdIDText.delete(0, tk.END)
            LabelingBirdIDText.config(fg='grey')
            LabelingBirdIDText.insert(0, "Bird ID")

    def labeling_handle_focus_in_2(_):
        if LabelingBirdIDText2.get() == "Bird ID":
            LabelingBirdIDText2.delete(0, tk.END)
            LabelingBirdIDText2.config(fg='black')

    def labeling_handle_focus_out_2(_):
        if LabelingBirdIDText2.get() == "":
            LabelingBirdIDText2.delete(0, tk.END)
            LabelingBirdIDText2.config(fg='grey')
            LabelingBirdIDText2.insert(0, "Bird ID")

    def syntax_focus_in(_):
        if SyntaxBirdID.get() == "Bird ID":
            SyntaxBirdID.delete(0, tk.END)
            SyntaxBirdID.config(fg='black')

    def syntax_focus_out(_):
        if SyntaxBirdID.get() == "":
            SyntaxBirdID.delete(0, tk.END)
            SyntaxBirdID.config(fg='grey')
            SyntaxBirdID.insert(0, "Bird ID")

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

    def LabelingFileExplorerFunction():
        global segmentations_path
        global Bird_ID
        segmentations_path = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")]) #e.g. "C:/where_my_segmentation_table_is/Bird_ID_segmentations.csv"
        NameLength = 45
        if len(segmentations_path) > NameLength:
            segmentations_path_short = segmentations_path[0:NameLength-(len(segmentations_path.split("/")[-1])+1)]+".../"+segmentations_path.split("/")[-1]
            LabelingFileDisplay = tk.Label(LabelingMainFrame,text=segmentations_path_short)
        else:
            LabelingFileDisplay = tk.Label(LabelingMainFrame,text=segmentations_path)

        LabelingFileDisplay.grid(row=1, column=1, sticky="w")
        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(segmentations_path) != None:
            LabelingBirdIDText.delete(0, tk.END)
            LabelingBirdIDText.config(fg='black')
            LabelingBirdIDText.insert(0, str(pattern.search(segmentations_path).group()))
            Bird_ID = str(pattern.search(segmentations_path).group())

    def LabelingFileExplorerFunctionSaving():
        global labeling_path
        global Bird_ID
        labeling_path = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])  # e.g. "C:/where_my_segmentation_table_is/Bird_ID_segmentations.csv"
        NameLength = 45
        if len(labeling_path) > NameLength:
            labeling_path_short = labeling_path[
                                       0:NameLength - (len(labeling_path.split("/")[-1]) + 1)] + ".../" + \
                                       labeling_path.split("/")[-1]
            LabelingFileDisplay2 = tk.Label(LabelingBulkSave, text=labeling_path_short)
        else:
            LabelingFileDisplay2 = tk.Label(LabelingBulkSave, text=labeling_path)

        LabelingFileDisplay2.grid(row=1, column=1, sticky="w")
        LabelingFileDisplay2.grid(row=0, column=1, sticky="w")

        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(labeling_path) != None:
            LabelingBirdIDText2.delete(0, tk.END)
            LabelingBirdIDText2.config(fg='black')
            LabelingBirdIDText2.insert(0, str(pattern.search(labeling_path).group()))
            Bird_ID = str(pattern.search(labeling_path).group())

    def AcousticsFileExplorerFunction():
        global AcousticsDirectory
        global Bird_ID
        AcousticsDirectory = filedialog.askopenfilename(filetypes = [(".wav files", "*.wav")])
        AcousticsFileDisplay = tk.Label(AcousticsMainFrameSingle, text=AcousticsDirectory.split("/")[-1])
        AcousticsFileDisplay.grid(row=1, column=1, columnspan=2)

        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(AcousticsDirectory.split("/")[-2]) != None:
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')
            BirdIDText.insert(0, str(pattern.search(AcousticsDirectory.split("/")[-2]).group()))
            Bird_ID = str(pattern.search(AcousticsDirectory.split("/")[-2]).group())

    def SyntaxFileExplorer():
        global SyntaxDirectory
        SyntaxDirectory = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        SyntaxFileDisplay = tk.Label(SyntaxMainFrame, text=SyntaxDirectory.split("/")[-1])
        SyntaxFileDisplay.grid(row=1, column=1)

    def MultiAcousticsFileExplorerFunction():
        global MultiAcousticsDirectory
        global Bird_ID
        MultiAcousticsDirectory = filedialog.askopenfilename(filetypes = [(".csv files", "*.csv")])
        MultiAcousticsFileDisplay = tk.Label(AcousticsMainFrameMulti, text=MultiAcousticsDirectory.split("/")[-1])
        MultiAcousticsFileDisplay.grid(row=1, column=1, columnspan=2)

        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(MultiAcousticsDirectory.split("/")[-2]) != None:
            BirdIDText.delete(0, tk.END)
            BirdIDText.config(fg='black')
            BirdIDText.insert(0, str(pattern.search(MultiAcousticsDirectory.split("/")[-2]).group()))
            Bird_ID = str(pattern.search(MultiAcousticsDirectory.split("/")[-2]).group())

    def SegmentButtonClick():
        SegmentingProgressLabel = tk.Label(SegmentationFrame, text="Segmenting...", font=("Arial", 10), justify=CENTER)
        SegmentingProgressLabel.grid(row=7, column=0, columnspan=2)
        global Bird_ID
        Bird_ID = BirdIDText.get()
        global song_folder
        song_folder = song_folder + "/"
        global segmenter
        segmenter = avn.segmentation.MFCCDerivative()

        # Default upper and lower thresholds are 0.1 and -0.1 respectively
        try:
            MaxThreshold= float(MaxThresholdText.get())
        except:
            MaxThreshold=0.1
        try:
            MinThreshold= float(MinThresholdText.get())
        except:
            MinThreshold=-0.1
        global seg_data
        seg_data = segmenter.make_segmentation_table(Bird_ID, song_folder,
            upper_threshold=MaxThreshold,
            lower_threshold=MinThreshold)

        out_file_dir = song_folder
        SegmentingProgressLabel.config(text="Segmentation Complete!")
        SegmentingProgressLabel.update()
        print("Segmentation Preview:")
        print(seg_data.seg_table.head())
        try:
            seg_data.save_as_csv(out_file_dir)
            print("Successfully saved segmentation data!")
        except:
            print("Failed to save segmentation data!")
        try:
            shutil.rmtree(song_folder + "/TempSpectrogramFiles/")
        except:
            #print("Directory not found: "+song_folder + "/TempSpectrogramFiles/")
            pass
        try:
            os.remove(str(song_folder) + 'TempFig.png')
        except:
            #print("Directory not found: "+song_folder + "TempFig.png")
            pass

    def SpectrogramDisplay(Direction, FolderSize = 20):
        global ImgLabel
        global FileID
        global fig
        global Bird_ID
        global song_folder
        global newfilelist
        global seg_data
        global SpectrogramWindow

        # Deletes previously loaded spectrogram image
        try:
            ImgLabel.destroy()
        except: pass

        # I break song folder into chunks to make it quicker to process spectrograms for viewing
        if Direction == "Left":
            FileID = FileID-1
        if Direction == "Right":
            FileID = FileID+1
        if Direction == "Start":
            FileID = 0
        if FileID == 0:
            filelist = glob.glob(str(song_folder) + "/*.wav")
            newfilelist = []
            for file in filelist:
                temp1, temp2 = file.split("\\")
                newfilelist.append(temp1 + "/" + temp2)
        if len(newfilelist) >= FolderSize:
            if FileID % FolderSize == 0:
                segFolderCount = int(FileID // FolderSize)
            else:
                segFolderCount = int((FileID // FolderSize) + 1)

            # Make directory to put temporary files such as spectrogram .png files and .wav files
            try:
                os.makedirs(song_folder + "/TempSpectrogramFiles/")
            except: pass

            # Creates subfolders in TempSpectrogramFiles folder for each chunk of .wav files
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
        else:
            segFolderCombined = song_folder+"/"

        # Retrieve thresholds from gui -- default upper and lower thresholds are 0.1 and -0.1 respectively
        try:
            upper_threshold = float(MaxThresholdText.get())
        except:
            upper_threshold = 0.1
        try:
            lower_threshold = float(MinThresholdText.get())
        except:
            lower_threshold = -0.1

        # Make spectrograms
        segmenter = avn.segmentation.MFCCDerivative()
        Temp_seg_data = segmenter.make_segmentation_table(Bird_ID, segFolderCombined,
                                                     upper_threshold=upper_threshold,
                                                     lower_threshold=lower_threshold)

        #fig, ax, ax2, x_axis, spectrogram, axUpper, axLower, UT, LT = avn.segmentation.Plot.plot_seg_criteria(Temp_seg_data, segmenter,
        #                                                                        "MFCC Derivative",
        #                                                                        file_idx=FileID, figsize = (10,2.5),
        #                                                                        upper_threshold=upper_threshold, lower_threshold=lower_threshold)


        Temp_seg_data.seg_table['labels'] = ""
        #print(Temp_seg_data.seg_table)
        fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(Temp_seg_data.seg_table, song_folder, "",
                                                                            song_file_index=FileID, figsize=(12, 4),
                                                                            fontsize=14, add_legend=False)

        # Save spectrogram




        #fig.savefig(str(song_folder)+"/"+'TempFig.png')
        #print("Figure Saved:"+'TempFig.png')
        #img = PhotoImage(file=str(song_folder)+"/"+'TempFig.png')

        # Create pop-up window to display spectrograms -- only needed to be created once and is reused for subsequent spectrograms
        def SaveSegDisplay(fig, newfilelist, FileID):
            SaveMessage = tk.Label(SpectrogramWindow, text="Saving Spectrogram...", justify="center")
            SaveMessage.grid(row=1, column=50, columnspan=3)
            fig.savefig(str(newfilelist[FileID]) + ".png")
            time.sleep(0.1)
            SaveMessage.config(text="Spectrogram Saved!")
            SaveMessage.update()

        if FileID == 0:
            SpectrogramWindow = tk.Toplevel(gui)
            LeftButton = tk.Button(SpectrogramWindow, command=lambda: SpectrogramDisplay("Left"), text="Previous")
            LeftButton.grid(row=0, column=50)
            RightButton = tk.Button(SpectrogramWindow, command=lambda: SpectrogramDisplay("Right"), text="Next")
            RightButton.grid(row=0, column=51)
            SaveButton = tk.Button(SpectrogramWindow, command=lambda: SaveSegDisplay(fig, newfilelist, FileID), text="Save")
            SaveButton.grid(row=0, column=52)

        # Load spectrogram into gui
        #ImgLabel = tk.Label(SpectrogramWindow, image=img)
        #ImgLabel.grid(row=2, columnspan=100)
        #ImgLabel.update()

        canvas = FigureCanvasTkAgg(fig, master=SpectrogramWindow)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, columnspan=100)

        # Update gui so it displays current spectrogram
        SpectrogramWindow.mainloop()

### All code for labeling comes from Single_Directory_Template_for_AVN_labeling file in locally installed avn files
    # located at C:\Users\ethan\Desktop\Roberts_Lab_Files\AVNGUI\Local_AVN_Installation_Materials
    def labelingPlaceholder():
        LabelingProgress = tk.Label(LabelingFrame, text="Labeling...", justify="center")
        LabelingProgress.grid(row=6, column=0, columnspan=2)
        time.sleep(1)
        global segmentations_path
        global Bird_ID
        global song_folder
        global LabelingFileDisplay
        segmentations_path_temp = segmentations_path #e.g. "C:/where_my_wav_files_are/Bird_ID/90/"
        tempsong_folder = segmentations_path.split("/")
        song_folder = ""
        for f in tempsong_folder[0:-1]:
            song_folder = song_folder+f+"/"
        output_file = song_folder+"/"+str(Bird_ID)+"_labels.csv" #e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"
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
        min_level_db = -28 # ERROR! If a .wav file contains a sound that's too quiet, the audio.values variable for that row will be NULL, causing an error
        win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
        hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
        n_fft = 512
        K = 10
        min_cluster_prop = 0.04
        embedding_dim = 2
        #############################################
        # load segmentations
        predictions_reformat = pd.read_csv(segmentations_path)

        # Find any syllables with a duration less than 0.025s, and add on 0.01ms before the onset and 0.01ms after the offset
        predictions_reformat['onsets'][predictions_reformat['offsets'] - predictions_reformat['onsets'] < 0.025] = \
        predictions_reformat['onsets'] - 0.01


        # 8	2.2407256235827666	2.2523356009070294	B651_44571.26718440_1_10_7_25_18.wav

        segmentations = predictions_reformat

        # sometimes this padding can create negative onset times, which will cause errors.
        # correct this by setting any onsets <0 to 0.
        segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

        # Add syllable audio to dataframe
        syllable_dfs = pd.DataFrame()

        for song_file in segmentations.files.unique():
            file_path = song_folder + "/" + song_file
            song = dataloading.SongFile(file_path)
            song.bandpass_filter(500, 15000)

            syllable_df = segmentations[segmentations['files'] == song_file]

            # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                    for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
            syllable_dfs = pd.concat([syllable_dfs, syllable_df])
            if song_file == 'B651_44571.26718440_1_10_7_25_18.wav':
                for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values):
                    print("----------------")
                    print(st)
                    print(et)
                    print(song.sample_rate)
                    print(song.data)
                    print(song.data[int(st * song.sample_rate): int(et * song.sample_rate)])

        # Normalize the audio  --- Ethan's comment: This won't work when there's an empty array for syllable_dfs_audio_values, so I'm just going to set those to '[0]'
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


        global hdbscan_df
        hdbscan_df = syllable_dfs
        hdbscan_df["labels"] = clusterer.labels_

        hdbscan_df["X"] = embedding[:, 0]
        hdbscan_df["Y"] = embedding[:, 1]
        hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")

        hdbscan_df.to_csv(output_file)
        ################################################
        LabelingScatterplot = sns.scatterplot(data=hdbscan_df, x="X", y="Y", hue="labels", alpha=0.25, s=5)
        plt.title("My Bird's Syllables");
        LabelingFig = LabelingScatterplot.get_figure()

        LabelingFig.savefig(song_folder+"/LabelingClusters.png")
        print("Figure Saved:" + 'LabelingClusters.png')
        LabelingImg = PhotoImage(file='LabelingClusters.png')

        LabelingWindow = tk.Toplevel(gui)

        LabelingImgLabel = tk.Label(LabelingWindow, image=LabelingImg)
        LabelingImgLabel.grid(columnspan=2)
        LabelingImgLabel.update()
        song_folder = song_folder[0:-1]
        LabelingWindow.update()
        print("UMAP Generated!")

        LabelingDisplay()
        # Make folder for storing labeling photos

    def LabelingDisplay(Direction = "Start"):
        global label_FileID
        global song_folder
        global hdbscan_df
        if Direction == "Start":
            try:
                os.makedirs(song_folder + "/LabelingPhotos")
            except: pass
            LabelingDisplayWindow = tk.Toplevel(gui)
            LabelingPreviousButton = tk.Button(LabelingDisplayWindow, text="Previous", command=lambda: LabelingDisplay(Direction = "Left"))
            LabelingPreviousButton.grid(row=0, column=10)
            LabelingNextButton = tk.Button(LabelingDisplayWindow, text="Next", command= lambda: LabelingDisplay(Direction = "Right"))
            LabelingNextButton.grid(row=0, column=11)
            LabelingSaveButton = tk.Button(LabelingDisplayWindow, text="Save", command=lambda: LabelingDisplaySave)
            LabelingSaveButton.grid(row=0, column=12)
            LabelingDisplayFrame = tk.Frame(LabelingDisplayWindow)
            LabelingDisplayFrame.grid(row=1, column=0, columnspan=21)
            label_FileID = 0
        if Direction == "Left":
            label_FileID -= 1
        if Direction == "Right":
            label_FileID += 1

        label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                            song_file_index=label_FileID, figsize=(12, 4),
                                                                            fontsize=14)

        label_canvas = FigureCanvasTkAgg(label_fig, master=LabelingDisplayFrame)  # A tk.DrawingArea.
        label_canvas.draw()
        label_canvas.get_tk_widget().grid()

        LabelingDisplayWindow.mainloop()

    def LabelingDisplaySave():
        global label_FileID
        global song_folder
        global segmentations_path
        hdbscan_df = pd.read_csv(segmentations_path)
        fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                                  song_file_index=label_FileID,
                                                                                  figsize=(15, 4),
                                                                                  fontsize=14)
        fig.savefig(str(song_folder) + "/LabelingPhotos/" + str(song_file_name) + ".png")
        print("Fig saved as " + str(song_file_name) + "!")

    def LabelingSavePhotos():
        global LabelingSaveAllCheck
        global label_FileID
        global labeling_path
        global BulkLabelFiles_to_save
        song_folder = ""
        for i in labeling_path.split("/")[:-1]:
            song_folder = song_folder+i+"/"
        hdbscan_df = pd.read_csv(labeling_path)
        LabelingSaveProgress = tk.Label(LabelingBulkSave, text="Saving...")
        LabelingSaveProgress.grid(row=6, column=0, columnspan=2)
        time.sleep(0.1)
        try:
            os.makedirs(song_folder+"/LabeingPhotos")
        except: pass
        if LabelingSaveAllCheck.get() == 1:
            i = glob.glob(str(song_folder) + "/*.wav")
            for a in range(len(i)):
                LabelingSaveProgress.config(
                    text="Saving Labeling Images (" + str(a + 1) + " of " + str(len(i)) + ")")
                LabelingSaveProgress.update()
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                                    song_file_index=a, figsize=(20, 5),
                                                                                    fontsize=14)
                fig.savefig(str(song_folder) + "/LabelingPhotos/" + str(song_file_name) + ".png")
        if LabelingSaveAllCheck.get() == 0:
            for a in BulkLabelFiles_to_save:
                LabelingSaveProgress.config(
                    text="Saving Labeling Images (" + str(BulkLabelFiles_to_save.index(a) + 1) + " of " + str(len(BulkLabelFiles_to_save)) + ")")
                LabelingSaveProgress.update()
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                                    song_file_index=BulkLabelFiles_to_save.index(a), figsize=(20, 5),
                                                                                    fontsize=14)
                fig.savefig(str(song_folder) + "/LabelingPhotos/" + str(song_file_name) + ".png")
        LabelingSaveProgress.config(text="Labeling Complete!")
        LabelingSaveProgress.update()
        time.sleep(5)
        LabelingSaveProgress.destroy()

    def AcousticsFeaturesOneFile():
        AcousticsProgress = tk.Label(AcousticsFrame, text="Calculating Acoustics...")
        AcousticsProgress.grid(row=12, column=1)
        global AcousticsDirectory
        song = dataloading.SongFile(AcousticsDirectory)
        song_interval = acoustics.SongInterval(song, onset=0, offset=2)
        MasterFeatureList = ['Goodness', 'Mean_frequency', 'Entropy', 'Amplitude', 'Amplitude_modulation', 'Frequency_modulation', 'Pitch']
        FeatureList = []
        if RunGoodness.get() == 1:
            FeatureList.append("Goodness")
        if RunMean_frequency.get() == 1:
            FeatureList.append("Mean_frequency")
        if RunEntropy.get() == 1:
            FeatureList.append("Entropy")
        if RunAmplitude.get() == 1:
            FeatureList.append("Amplitude")
        if RunAmplitude_modulation.get() == 1:
            FeatureList.append("Amplitude_modulation")
        if RunFrequency_modulation.get() == 1:
            FeatureList.append("Frequency_modulation")
        if RunPitch.get() == 1:
            FeatureList.append("Pitch")
        #features = song_interval.calc_all_features(features=FeatureList)
        #feature_stats = song_interval.calc_feature_stats()
        AcousticsOutputDirectory = ''
        for i in AcousticsDirectory.split("/")[:-1]:
            AcousticsOutputDirectory = AcousticsOutputDirectory+i+"/"
        song_interval.save_features(AcousticsOutputDirectory, str(AcousticsDirectory.split("/")[-1][:-4]))
        song_interval.save_feature_stats(AcousticsOutputDirectory, "Stats__"+str(AcousticsDirectory.split("/")[-1][:-4]))
        AcousticsProgress.config(text="Acoustics Calculations Complete!")

    def AcousticsFeaturesManyFiles():
        MultiAcousticsProgress = tk.Label(MultiAcousticsFrame, text="Calculating Acoustics...")
        MultiAcousticsProgress.grid(row=12, column=1)
        time.sleep(1)
        MultiAcousticsProgress.update()
        global MultiAcousticsDirectory
        global Bird_ID
        syll_df = pd.read_csv(str(MultiAcousticsDirectory))
        MultiAcousticsOutputDirectory = ""
        for i in MultiAcousticsDirectory.split("/")[:-1]:
            MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory+i+"/"

        # Try below function, assuming BirdID is chosen, otherwise grab bird ID from directory and proceed
        try:
            acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=MultiAcousticsOutputDirectory)
        except:
            pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
            if pattern.search(MultiAcousticsDirectory.split("/")[-1]) != None:
                Bird_ID = str(pattern.search(MultiAcousticsDirectory.split("/")[-2]).group())
            acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=MultiAcousticsOutputDirectory)

        FeatureList = []
        if MultiRunGoodness.get() == 1:
            FeatureList.append("Goodness")
        if MultiRunMean_frequency.get() == 1:
            FeatureList.append("Mean_frequency")
        if MultiRunEntropy.get() == 1:
            FeatureList.append("Entropy")
        if MultiRunAmplitude.get() == 1:
            FeatureList.append("Amplitude")
        if MultiRunAmplitude_modulation.get() == 1:
            FeatureList.append("Amplitude_modulation")
        if MultiRunFrequency_modulation.get() == 1:
            FeatureList.append("Frequency_modulation")
        if MultiRunPitch.get() == 1:
            FeatureList.append("Pitch")
        #features = acoustic_data.calc_all_features(features=FeatureList)

        acoustic_data.save_features(out_file_path=MultiAcousticsOutputDirectory,
                                         file_name=str(Bird_ID) + "_feature_table",
                                         features=FeatureList)
        acoustic_data.save_feature_stats(out_file_path=MultiAcousticsOutputDirectory, file_name=str(Bird_ID)+"_syll_table",
                                         features=FeatureList)
        MultiAcousticsProgress.config(text="Acoustics Calculations Complete!")

    def SyllableSyntax():
        import avn.syntax as syntax
        import avn.plotting as plotting
        global SyntaxDirectory
        global DropCalls
        SyntaxProgress = tk.Label(SyntaxFrame, text="Running...")
        SyntaxProgress.grid(row=5, column=0)
        Bird_ID = SyntaxBirdID.get()
        syll_df = pd.read_csv(SyntaxDirectory)
        syntax_data = syntax.SyntaxData(Bird_ID, syll_df)
        tempSyntaxDirectory = ""
        for a in SyntaxDirectory.split("/")[:-2]:
            tempSyntaxDirectory = tempSyntaxDirectory+a+"/"
        syntax_data.add_file_bounds(tempSyntaxDirectory)
        syntax_data.add_gaps(min_gap=0.2)
        gaps_df = syntax_data.get_gaps_df()
        if DropCalls.get() == 1:
            syntax_data.drop_calls()
        syntax_data.make_transition_matrix()
        print(syntax_data.trans_mat)
        entropy_rate = syntax_data.get_entropy_rate()
        entropy_rate_norm = entropy_rate / np.log2(len(syntax_data.unique_labels) + 1)
        prob_repetitions = syntax_data.get_prob_repetitions()
        single_rep_counts, single_rep_stats = syntax_data.get_single_repetition_stats()
        intro_notes_df = syntax_data.get_intro_notes_df()
        prop_sylls_in_short_bouts = syntax_data.get_prop_sylls_in_short_bouts(max_short_bout_len=2)
        per_syll_stats = syntax.Utils.merge_per_syll_stats(single_rep_stats, prop_sylls_in_short_bouts, intro_notes_df)
        pair_rep_counts, pair_rep_stats = syntax_data.get_pair_repetition_stats()
        tempSyntaxDirectory2 = ""
        for a in SyntaxDirectory.split("/")[:-1]:
            tempSyntaxDirectory2 = tempSyntaxDirectory2+a+"/"
        syntax_analysis_metadata = syntax_data.save_syntax_data(tempSyntaxDirectory2)
        SyntaxProgress.config(text="Complete!")

    def FindPlainSpectrograms():
        global PlainDirectory
        PlainDirectory = filedialog.askdirectory()
        PlainDirectoryLabel = tk.Label(PlainSpectro, text=str(PlainDirectory))
        PlainDirectoryLabel.grid(row=1, column=1)

    def FindPlainSpectrogramsAlt():
        global PlainDirectoryAlt
        PlainDirectoryAlt = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
        PlainDirectoryLabelAlt = tk.Label(PlainSpectroAlt, text=str(len(PlainDirectoryAlt))+" files selected")
        PlainDirectoryLabelAlt.grid(row=1, column=1)

    def PrintPlainSpectrograms():
        global PlainDirectory
        try:
            os.makedirs(PlainDirectory + "/Unlabeled Spectrograms")
        except: pass
        FileList = glob.glob(str(PlainDirectory) + "/*.wav")
        PlainProgressLabel = tk.Label(PlainSpectro, text="Generating Spectrograms...")
        PlainProgressLabel.grid(row=3, column=1)
        f = 0
        for file in FileList:
            f+=1
            PlainProgressLabel.config(text="Generating Spectrograms ("+str(f)+" of "+str(len(FileList))+")")
            PlainProgressLabel.update()
            time.sleep(0.1)
            song = avn.dataloading.SongFile(file)
            spectrogram_db = avn.plotting.make_spectrogram(song)
            plt = avn.plotting.plot_spectrogram(spectrogram_db, song.sample_rate) # Modified avn.plotting.plot_spectrograms to return plot

            plt.savefig(PlainDirectory+"/Unlabeled Spectrograms"+"/"+str(file.split("/")[-1].split("\\")[-1])+".png")

    def PrintPlainSpectrogramsAlt():
        global PlainDirectoryAlt
        PlainProgressLabel = tk.Label(PlainSpectroAlt, text="Generating Spectrograms...")
        PlainProgressLabel.grid(row=3, column=1)
        f = 0
        SaveLocation = ""
        for x in PlainDirectoryAlt[0].split("/")[:-1]:
            SaveLocation = SaveLocation+x+"/"
        for file in PlainDirectoryAlt:
            f+=1
            PlainProgressLabel.config(text="Generating Spectrograms ("+str(f)+" of "+str(len(PlainDirectoryAlt))+")")
            PlainProgressLabel.update()
            time.sleep(0.1)
            song = avn.dataloading.SongFile(file)
            spectrogram_db = avn.plotting.make_spectrogram(song)
            plt = avn.plotting.plot_spectrogram(spectrogram_db, song.sample_rate) # Modified avn.plotting.plot_spectrograms to return plot

            plt.savefig(SaveLocation+"/Unlabeled Spectrograms"+"/"+str(file.split("/")[-1].split("\\")[-1])+".png")
        PlainProgressLabel.config(text="Complete!")

    ### Initialize gui ###
    gui = tk.Tk()
    gui.title("AVN Segmentation and Labeling")

    ParentStyle = ttk.Style()
    ParentStyle.configure('Parent.TNotebook.Tab', font=('Arial', '10', 'bold'))

    notebook = ttk.Notebook(gui, style='Parent.TNotebook.Tab')
    notebook.grid()

    MasterFrameWidth = 500
    MasterFrameHeight = 300

    ### Segmentation Window ###
    SegmentationFrame = tk.Frame(gui)
    notebook.add(SegmentationFrame, text="Segmentation")

    SegmentationMainFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook = ttk.Notebook(SegmentationFrame)
    SegmentationNotebook.grid(row=1)
    SegmentationNotebook.add(SegmentationMainFrame, text="Home")
    SegmentationMainFrame.grid_propagate(False)
    SegmentationSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook.add(SegmentationSettingsFrame, text="Advanced Settings")
    SegmentationInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SegmentationNotebook.add(SegmentationInfoFrame, text="Info")


    BirdIDText = tk.Entry(SegmentationMainFrame, font=("Arial", 15), justify="center", fg="grey")
    BirdIDText.insert(0, "Bird ID")
    BirdIDText.bind("<FocusIn>", handle_focus_in)
    BirdIDText.bind("<FocusOut>", handle_focus_out)
    BirdIDText.grid(row=2, column=1, sticky="w")

    BirdIDLabel = tk.Label(SegmentationMainFrame, text="Bird ID:")
    BirdIDLabel.grid(row=2, column=0)

    FileExplorer = tk.Button(SegmentationMainFrame, text="Find Folder", command = lambda : FileExplorerFunction())
    FileExplorer.grid(row=1, column=0)

    SegmentButton = tk.Button(SegmentationMainFrame, text="Segment!", command = lambda : SegmentButtonClick())
    SegmentButton.grid(row=6, columnspan=2)

    MinThresholdText = tk.Entry(SegmentationMainFrame, justify="center", fg='grey', font=("Arial", 15))
    MinThresholdText.insert(0, "-0.1")
    MinThresholdText.grid(row=3, column=1, sticky="w")
    MinThresholdLabel = tk.Label(SegmentationMainFrame, text="Min Threshold:")
    MinThresholdText.bind("<FocusIn>", MinFocusIn)
    MinThresholdText.bind("<FocusOut>", MinFocusOut)
    MinThresholdLabel.grid(row=3, column=0, sticky="w")

    MaxThresholdText = tk.Entry(SegmentationMainFrame, justify="center", fg='grey', font=("Arial", 15))
    MaxThresholdText.insert(0, "0.1")
    MaxThresholdText.grid(row=4, column=1, sticky="w")
    MaxThresholdLabel = tk.Label(SegmentationMainFrame, text="Max Threshold:")
    MaxThresholdText.bind("<FocusIn>", MaxFocusIn)
    MaxThresholdText.bind("<FocusOut>", MaxFocusOut)
    MaxThresholdLabel.grid(row=4, column=0)

    SpectrogramButton = tk.Button(SegmentationMainFrame, text="Segmentation Preview", command = lambda : SpectrogramDisplay("Start"))
    SpectrogramButton.grid(row=5, columnspan=2)


    ### Labeling Window ###
    LabelingFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(LabelingFrame, text="Labeling")

    LabelingMainFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook = ttk.Notebook(LabelingFrame)
    LabelingNotebook.grid(row=1)
    LabelingNotebook.add(LabelingMainFrame, text="Home")
    LabelingBulkSave = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingBulkSave, text="Bulk Saving")
    LabelingSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingSettingsFrame, text="Advanced Settings")
    LabelingInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingInfoFrame, text="Info")

    LabelingButton = tk.Button(LabelingMainFrame, text="Labeling", command = lambda : labelingPlaceholder())
    LabelingButton.grid(row=5, column=0, columnspan=2)

    LabelingBirdIDText = tk.Entry(LabelingMainFrame, font=("Arial", 15), justify="center", fg="grey")
    LabelingBirdIDText.insert(0, "Bird ID")
    LabelingBirdIDText.bind("<FocusIn>", labeling_handle_focus_in)
    LabelingBirdIDText.bind("<FocusOut>", labeling_handle_focus_out)
    LabelingBirdIDText.grid(row=2, column=1, sticky="w")

    LabelingBirdIDLabel = tk.Label(LabelingMainFrame, text="Bird ID:")
    LabelingBirdIDLabel.grid(row=2, column=0)

    LabelingFileExplorer = tk.Button(LabelingMainFrame, text="Find Segmentation File", command=lambda: LabelingFileExplorerFunction())
    LabelingFileExplorer.grid(row=1, column=0)

    #LabelingPhotoText = tk.Label(LabelingBulkSave, text="Number of labeling photos to save:")
    #LabelingPhotoText.grid(row=2, column=0)

    #LabelingPhotoCount = tk.Spinbox(LabelingBulkSave, justify="center", font=("Arial",15),from_=0, to=20)
    #LabelingPhotoCount.grid(row=3, column=1, sticky="w")
    #LabelingPhotoCount.bind("<FocusIn>",LabelingCountFocusIn)
    #LabelingPhotoCount.bind("<FocusOut>",LabelingCountFocusOut)


    LabelingBirdIDText2 = tk.Entry(LabelingBulkSave, font=("Arial", 15), justify="center", fg="grey")
    LabelingBirdIDText2.insert(0, "Bird ID")
    LabelingBirdIDText2.bind("<FocusIn>", labeling_handle_focus_in_2)
    LabelingBirdIDText2.bind("<FocusOut>", labeling_handle_focus_out_2)
    LabelingBirdIDText2.grid(row=2, column=1, sticky="w")

    LabelingBirdIDLabel2 = tk.Label(LabelingBulkSave, text="Bird ID:")
    LabelingBirdIDLabel2.grid(row=2, column=0)

    LabelingFileExplorer2 = tk.Button(LabelingBulkSave, text="Find Labeling File",
                                     command=lambda: LabelingFileExplorerFunctionSaving())
    LabelingFileExplorer2.grid(row=0, column=0)

    global LabelingSaveAllCheck
    LabelingSaveAllCheck = IntVar()
    def LabelingSaveAllFilesText(LabelingSaveAllCheck):
        global Overwrite_BulkLabelFiles
        if LabelingSaveAllCheck.get() == 1:
            Overwrite_BulkLabelFiles = tk.Label(LabelingBulkSave, text="Whole Folder Selected")
            Overwrite_BulkLabelFiles.grid(row=1, column=1)
        if LabelingSaveAllCheck.get() == 0:
            try:
                Overwrite_BulkLabelFiles.destroy()
            except: pass

    LabelingSaveAll = tk.Checkbutton(LabelingBulkSave, text="Save Entire Folder", command=lambda:LabelingSaveAllFilesText(LabelingSaveAllCheck), variable=LabelingSaveAllCheck)
    LabelingSaveAll.grid(row=4, column=1)

    LabelingBulkSaveButton = tk.Button(LabelingBulkSave, text="Save", command=lambda:LabelingSavePhotos())
    LabelingBulkSaveButton.grid(row=5, column=0, columnspan=2)

    def BulkLablingFileText():
        global BulkLabelFiles_to_save
        BulkLabelFiles_to_save = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
        BulkLabelFiles = tk.Label(LabelingBulkSave, text=str(len(BulkLabelFiles_to_save)) + " files selected")
        BulkLabelFiles.grid(row=1, column=1)

    LabelingBulkFileSelect = tk.Button(LabelingBulkSave, text="Select Song Files to Save", command=lambda:BulkLablingFileText())
    LabelingBulkFileSelect.grid(row=1, column = 0)

    ### Single File Acoustic Features Window ###
    AcousticsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(AcousticsFrame, text="Acoustic Features")

    AcousticsMainFrameSingle = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook = ttk.Notebook(AcousticsFrame)
    AcousticsNotebook.grid(row=1)
    AcousticsNotebook.add(AcousticsMainFrameSingle, text="Single File")

    AcousticsFileExplorer = tk.Button(AcousticsMainFrameSingle, text="Find Song File",
                                      command=lambda: AcousticsFileExplorerFunction())
    AcousticsFileExplorer.grid(row=1, column=0)

    AcousticsText = tk.Label(AcousticsMainFrameSingle, text="Please select which acoustics features you would like to analyze:", justify="center")
    AcousticsText.grid(row=0, column=0, columnspan=2)

    global RunGoodness
    RunGoodness = IntVar()
    global RunMean_frequency
    RunMean_frequency = IntVar()
    global RunEntropy
    RunEntropy = IntVar()
    global RunAmplitude
    RunAmplitude = IntVar()
    global RunAmplitude_modulation
    RunAmplitude_modulation = IntVar()
    global RunFrequency_modulation
    RunFrequency_modulation = IntVar()
    global RunPitch
    RunPitch = IntVar()
    global CheckAll
    CheckAll = IntVar()

    def CheckAllBoxes(CheckAll):
        global RunGoodness
        global RunMean_frequency
        global RunEntropy
        global RunAmplitude
        global RunAmplitude_modulation
        global RunFrequency_modulation
        global RunPitch
        ButtonNameList = [RunGoodness,RunMean_frequency,RunEntropy,RunAmplitude,RunAmplitude_modulation,RunFrequency_modulation,RunPitch]

        if CheckAll.get() == 1:
            for checkbox in ButtonNameList:
                checkbox.set(1)
        if CheckAll.get() == 0:
            for checkbox in ButtonNameList:
                checkbox.set(0)

    CheckAll_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text='Select All', variable=CheckAll, command=lambda:CheckAllBoxes(CheckAll))
    CheckAll_CheckBox.grid(row=2, column=1, columnspan=1)
    Goodness_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text= "Goodness", anchor=tk.W, variable=RunGoodness)
    Goodness_CheckBox.grid(row=3, column=1, columnspan=1)
    Mean_frequency_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Mean Frequency",anchor=tk.W, variable=RunMean_frequency)
    Mean_frequency_CheckBox.grid(row=4, column=1, columnspan=1)
    Entropy_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Entropy", anchor=tk.W, variable=RunEntropy)
    Entropy_CheckBox.grid(row=5, column=1, columnspan=1)
    Amplitude_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Amplitude", anchor=tk.W, variable=RunAmplitude)
    Amplitude_CheckBox.grid(row=6, column=1, columnspan=1)
    Amplitude_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Amplitude Modulation", anchor=tk.W, variable=RunAmplitude_modulation)
    Amplitude_modulation_CheckBox.grid(row=7, column=1, columnspan=1)
    Frequency_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Frequency Modulation", anchor=tk.W, variable=RunFrequency_modulation)
    Frequency_modulation_CheckBox.grid(row=8, column=1, columnspan=1)
    Pitch_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Pitch", anchor=tk.W, variable=RunPitch)
    Pitch_CheckBox.grid(row=9, column=1, columnspan=1)
    AcousticsRunButton = tk.Button(AcousticsMainFrameSingle, text="Run", command=lambda: AcousticsFeaturesOneFile())
    AcousticsRunButton.grid(row=10, column=1)

    ### Multiple File Acoustic Features Window ###
    #MultiAcousticsFrame = tk.Frame()
    #notebook.add(MultiAcousticsFrame, text="Acoustic Features - Multiple Files")

    AcousticsMainFrameMulti = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsMainFrameMulti, text="Multiple Files")
    AcousticsSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsSettingsFrame, text="Advanced Settings")
    AcousticsInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsInfoFrame, text="Info")


    MultiAcousticsFileExplorer = tk.Button(AcousticsMainFrameMulti, text="Find Segmentation File",command=lambda: MultiAcousticsFileExplorerFunction())
    MultiAcousticsFileExplorer.grid(row=1, column=0)
    MultiAcousticsText = tk.Label(AcousticsMainFrameMulti, text="Please select which acoustics features you would like to analyze:")
    MultiAcousticsText.grid(row=0, column=0, columnspan=3)

    global MultiRunGoodness
    MultiRunGoodness = IntVar()
    global MultiRunMean_frequency
    MultiRunMean_frequency = IntVar()
    global MultiRunEntropy
    MultiRunEntropy = IntVar()
    global MultiRunAmplitude
    MultiRunAmplitude = IntVar()
    global MultiRunAmplitude_modulation
    MultiRunAmplitude_modulation = IntVar()
    global MultiRunFrequency_modulation
    MultiRunFrequency_modulation = IntVar()
    global MultiRunPitch
    MultiRunPitch = IntVar()
    global MultiCheckAll
    MultiCheckAll = IntVar()

    def MultiCheckAllBoxes(CheckAll):
        global MultiRunGoodness
        global MultiRunMean_frequency
        global MultiRunEntropy
        global MultiRunAmplitude
        global MultiRunAmplitude_modulation
        global MultiRunFrequency_modulation
        global MultiRunPitch
        MultiButtonNameList = [MultiRunGoodness, MultiRunMean_frequency, MultiRunEntropy, MultiRunAmplitude, MultiRunAmplitude_modulation,
                          MultiRunFrequency_modulation, MultiRunPitch]

        if MultiCheckAll.get() == 1:
            for checkbox in MultiButtonNameList:
                checkbox.set(1)
        if MultiCheckAll.get() == 0:
            for checkbox in MultiButtonNameList:
                checkbox.set(0)

    MultiCheckAll_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text='Select All', variable=MultiCheckAll,
                                       command=lambda: MultiCheckAllBoxes(MultiCheckAll))
    MultiCheckAll_CheckBox.grid(row=2, column=1, columnspan=1)
    MultiGoodness_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Goodness", anchor=tk.W, variable=MultiRunGoodness)
    MultiGoodness_CheckBox.grid(row=3, column=1, columnspan=1)
    MultiMean_frequency_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Mean Frequency", anchor=tk.W,
                                             variable=MultiRunMean_frequency)
    MultiMean_frequency_CheckBox.grid(row=4, column=1, columnspan=1)
    MultiEntropy_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Entropy", anchor=tk.W, variable=MultiRunEntropy)
    MultiEntropy_CheckBox.grid(row=5, column=1, columnspan=1)
    MultiAmplitude_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Amplitude", anchor=tk.W, variable=MultiRunAmplitude)
    MultiAmplitude_CheckBox.grid(row=6, column=1, columnspan=1)
    MultiAmplitude_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Amplitude Modulation", anchor=tk.W,
                                                   variable=MultiRunAmplitude_modulation)
    MultiAmplitude_modulation_CheckBox.grid(row=7, column=1, columnspan=1)
    MultiFrequency_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Frequency Modulation", anchor=tk.W,
                                                   variable=MultiRunFrequency_modulation)
    MultiFrequency_modulation_CheckBox.grid(row=8, column=1, columnspan=1)
    MultiPitch_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Pitch", anchor=tk.W, variable=MultiRunPitch)
    MultiPitch_CheckBox.grid(row=9, column=1, columnspan=1)
    MultiAcousticsRunButton = tk.Button(AcousticsMainFrameMulti, text="Run", command=lambda: AcousticsFeaturesManyFiles())
    MultiAcousticsRunButton.grid(row=10, column=1)

    #SegmentationHelpButton = tk.Button(SegmentationFrame, text="Help", command= lambda:HelpButton)
    #SegmentationHelpButton.grid(row=0, column=5, sticky="e")

    SyntaxFrame = tk.Frame(gui)
    notebook.add(SyntaxFrame, text="Syntax")

    SyntaxMainFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SyntaxNotebook = ttk.Notebook(SyntaxFrame)
    SyntaxNotebook.grid(row=1)
    SyntaxNotebook.add(SyntaxMainFrame, text="Home")
    SyntaxSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SyntaxNotebook.add(SyntaxSettingsFrame, text="Advanced Settings")
    SyntaxInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    SyntaxNotebook.add(SyntaxInfoFrame, text="Info")

    SyntaxFileButton = tk.Button(SyntaxMainFrame, text="Select Labeling File", command=lambda:SyntaxFileExplorer())
    SyntaxFileButton.grid(row=1,column=0)
    SyntaxRunButton = tk.Button(SyntaxMainFrame, text="Run", command=lambda: SyllableSyntax())
    SyntaxRunButton.grid(row=4)
    SyntaxBirdID = tk.Entry(SyntaxMainFrame, fg="grey", font=("Arial",15), justify="center")
    SyntaxBirdID.insert(0, "Bird ID")
    SyntaxBirdID.bind("<FocusIn>", syntax_focus_in)
    SyntaxBirdID.bind("<FocusOut>", syntax_focus_out)
    SyntaxBirdID.grid(row=2)
    global DropCalls
    DropCalls = IntVar()
    DropCallsCheckbox = tk.Checkbutton(SyntaxMainFrame, text= "Drop Calls", variable=DropCalls)
    DropCallsCheckbox.grid(row=3)


    ### Plain Spectrogram Generation - Whole File ###
    PlainSpectro = tk.Frame(gui)
    notebook.add(PlainSpectro, text="Plain Spectrograms")
    
    PlainMainFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook = ttk.Notebook(PlainSpectro)
    PlainNotebook.grid(row=1)
    PlainNotebook.add(PlainMainFrame, text="Whole Folder")

    
    PlainFileExplorer = tk.Button(PlainMainFrame, text="Select Folder", command=lambda: FindPlainSpectrograms())
    PlainFileExplorer.grid(row=1, column=0)
    PlainSpectroRun = tk.Button(PlainMainFrame, text="Create Blank Spectrograms", command=lambda:PrintPlainSpectrograms())
    PlainSpectroRun.grid(row=2, column=0, columnspan=2)

    ### Plain Spectrogram Generation - Selected Files ###
    PlainSpectroAlt= tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainSpectroAlt, text="Selected Files")
    PlainFileExplorerAlt = tk.Button(PlainSpectroAlt, text="Select Files", command=lambda: FindPlainSpectrogramsAlt())
    PlainFileExplorerAlt.grid(row=1, column=0)
    PlainSpectroRunAlt = tk.Button(PlainSpectroAlt, text="Create Blank Spectrograms",
                                command=lambda: PrintPlainSpectrogramsAlt())
    PlainSpectroRunAlt.grid(row=2, column=0, columnspan=2)
    PlainSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainSettingsFrame, text="Advanced Settings")
    PlainInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainInfoFrame, text="Info")
    '''
    def BadFileTesting():
        def make_spec2(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
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
        predictions_reformat = pd.read_csv('C:/Users/ethan/Desktop/169/B651-0.1~0.1_seg_table.csv')
        BadSongPath = 'C:/Users/ethan/Desktop/Bad Song File - B651/B651_44571.26718440_1_10_7_25_18.wav'
        # Find any syllables with a duration less than 0.025s, and add on 0.01ms before the onset and 0.01ms after the offset
        predictions_reformat['onsets'][predictions_reformat['offsets'] - predictions_reformat['onsets'] < 0.025] = \
        predictions_reformat['onsets'] - 0.01

        segmentations = predictions_reformat

        # sometimes this padding can create negative onset times, which will cause errors.
        # correct this by setting any onsets <0 to 0.
        segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

        # Add syllable audio to dataframe
        syllable_dfs = pd.DataFrame()

        file_path = BadSongPath
        song = dataloading.SongFile(file_path)
        song.bandpass_filter(100, 15000)

        syllable_df = segmentations[segmentations['files'] == song_file]

        # this section is based on avgn.signalprocessing.create_spectrogram_dataset.get_row_audio()
        syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
        syllable_dfs = pd.concat([syllable_dfs, syllable_df])
    '''
    #print(ttk.Style().theme_names())
    ttk.Style().theme_use("clam")

    #BadFileTesting = tk.Button(LabelingMainFrame, text="Test Bad File", BadFileTesting)

    gui.mainloop()

except Exception:
    print(Exception)