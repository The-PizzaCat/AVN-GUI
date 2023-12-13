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
    from matplotlib.figure import Figure


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
        AcousticsDirectory = filedialog.askopenfilename(C)
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
        global Bird_ID
        SyntaxDirectory = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        SyntaxFileDisplay = tk.Label(SyntaxMainFrame, text=SyntaxDirectory.split("/")[-1])
        SyntaxFileDisplay.grid(row=1, column=1)

        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
        if pattern.search(SyntaxDirectory.split("/")[-1]) != None:
            SyntaxBirdID.delete(0, tk.END)
            SyntaxBirdID.config(fg='black')
            SyntaxBirdID.insert(0, str(pattern.search(SyntaxDirectory.split("/")[-1]).group()))
            Bird_ID = str(pattern.search(SyntaxDirectory.split("/")[-1]).group())

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
            song.bandpass_filter(int(bandpass_lower_cutoff_entry_Labeling.get()), int(bandpass_upper_cutoff_entry_Labeling.get()))

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
                                      hop_length=int(hop_length_entry_Labeling.get()),
                                      win_length=int(win_length_entry_Labeling.get()),
                                      n_fft=int(n_fft_entry_Labeling.get()),
                                      ref_db=int(ref_db_entry_Labeling.get()),
                                      amin=float(a_min_entry_Labeling.get()),
                                      min_level_db=int(min_level_db_entry_Labeling.get()))
                if syllable_spec.shape[1] > int(max_spec_size_entry_Labeling.get()):
                    print("Long Syllable Corrections! Spectrogram Duration = " + str(syllable_spec.shape[1]))
                    syllable_spec = syllable_spec[:, :int(max_spec_size_entry_Labeling.get())]

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

        # Generate Metadata File for Advanced Settings #
        LabelingSettingsMetadata = pd.DataFrame()
        Labeling_SettingsNames = ["bandpass_lower_cutoff_entry_Labeling","bandpass_upper_cutoff_entry_Labeling","a_min_entry_Labeling","ref_db_entry_Labeling","min_level_db_entry_Labeling","n_fft_entry_Labeling","win_length_entry_Labeling","hop_length_entry_Labeling","max_spec_size_entry_Labeling"]
        LabelingSettingsMetadata["Main"] = Labeling_SettingsNames

        Labeling_SettingsValues = pd.DataFrame({'Value':[bandpass_lower_cutoff_entry_Labeling.get(),bandpass_upper_cutoff_entry_Labeling.get(),a_min_entry_Labeling.get(),ref_db_entry_Labeling.get(),min_level_db_entry_Labeling.get(),n_fft_entry_Labeling.get(),win_length_entry_Labeling.get(),hop_length_entry_Labeling.get(),max_spec_size_entry_Labeling.get()]})
        LabelingUMAP_SettingsNames = pd.DataFrame({'UMAP':['n_neighbors_entry_Labeling','n_components_entry_Labeling','min_dist_entry_Labeling','spread_entry_Labeling','metric_entry_Labeling','random_state_entry_Labeling']})
        LabelingUMAP_SettingsValues = pd.DataFrame({'UMAP Value':        [n_neighbors_entry_Labeling.get(),n_components_entry_Labeling.get(),min_dist_entry_Labeling.get(),spread_entry_Labeling.get(),metric_variable.get(),random_state_entry_Labeling.get()]})
        LabelingClustering_SettingsNames = pd.DataFrame({'Clustering':['min_cluster_prop_entry_Labeling','spread_entry_Labeling']})
        LabelingClustering_SettingsValues = pd.DataFrame({'Clustering Values':[min_cluster_prop_entry_Labeling.get(), spread_entry_Labeling.get()]})

        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, Labeling_SettingsValues], axis=1)
        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingUMAP_SettingsNames], axis=1)
        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingUMAP_SettingsValues], axis=1)
        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingClustering_SettingsNames], axis=1)
        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingClustering_SettingsValues], axis=1)

        LabelingSettingsMetadata.to_csv(song_folder+"/LabelingSettings_Metadata.csv")

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
        AcousticsProgress = tk.Label(AcousticsMainFrameSingle, text="Calculating Acoustics...")
        AcousticsProgress.grid(row=12, column=1)
        global AcousticsDirectory
        song = dataloading.SongFile(AcousticsDirectory)
        if AcousticsOffset.get() == "End of File":
            song_interval = acoustics.SongInterval(song, onset=int(AcousticsOnset.get()), offset=None)
        else:
            song_interval = acoustics.SongInterval(song, onset=int(AcousticsOnset.get()), offset=int(AcousticsOffset.get()))
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
        MultiAcousticsProgress = tk.Label(AcousticsMainFrameMulti, text="Calculating Acoustics...")
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
            acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=MultiAcousticsOutputDirectory,
                                                   win_length=int(win_length_entry_Acoustics.get()),hop_length= int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()), max_F0=int(max_F0_entry_Acoustics.get()),min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=int(freq_range_entry_Acoustics.get()),baseline_amp=int(baseline_amp_entry_Acoustics.get()),fmax_yin=int(fmax_yin_entry_Acoustics.get()))
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
        # Generate Metadata File for Advanced Settings #
        AcousticSettingsMetadata = pd.DataFrame()
        Acoustic_SettingsNames = ["win_length_entry_Acoustics", "hop_length_entry_Acoustics",
                                  "n_fft_entry_Acoustics", "max_F0_entry_Acoustics", "min_frequency_entry_Acoustics",
                                  "freq_range_entry_Acoustics", "baseline_amp_entry_Acoustics", "fmax_yin_entry_Acoustics"]
        AcousticSettingsMetadata["Main"] = Acoustic_SettingsNames

        Acoustic_SettingsValues = pd.DataFrame({'Value': [win_length_entry_Acoustics.get(), hop_length_entry_Acoustics.get(), n_fft_entry_Acoustics.get(), max_F0_entry_Acoustics.get(), min_frequency_entry_Acoustics.get(), freq_range_entry_Acoustics.get(), baseline_amp_entry_Acoustics.get(), fmax_yin_entry_Acoustics.get()]})

        AcousticSettingsMetadata = pd.concat([AcousticSettingsMetadata, Acoustic_SettingsValues], axis=1)
        AcousticSettingsMetadata.to_csv(MultiAcousticsOutputDirectory + "/AcousticSettings_Metadata.csv")

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
        global syntax_data
        syntax_data = syntax.SyntaxData(Bird_ID, syll_df)
        tempSyntaxDirectory = ""
        for a in SyntaxDirectory.split("/")[:-2]:
            tempSyntaxDirectory = tempSyntaxDirectory+a+"/"
        syntax_data.add_file_bounds(tempSyntaxDirectory)
        syntax_data.add_gaps(min_gap=float(min_gap_entry_Syntax.get()))
        gaps_df = syntax_data.get_gaps_df()
        if DropCalls.get() == 1:
            syntax_data.drop_calls()
        syntax_data.make_transition_matrix()
        #print(syntax_data.trans_mat)
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

    def FindPlainSpectrogramsFolder():
        global PlainDirectory
        PlainDirectory = filedialog.askdirectory()
        PlainDirectoryLabel = tk.Label(PlainSpectroAlt, text=str(PlainDirectory))
        PlainDirectoryLabel.grid(row=1, column=1)

    def FindPlainSpectrogramsFiles():
        global PlainDirectoryAlt
        PlainDirectoryAlt = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
        PlainDirectoryLabelAlt = tk.Label(PlainMainFrame, text=str(len(PlainDirectoryAlt))+" files selected")
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

    def MoreInfo(Event):
        #print(Event.widget)
        if str(Event.widget) == ".!frame15.!button2": #win_length
            AcousticSettingsDialog.config(text="Length of window over which to calculate each \n feature in samples")
            AcousticsSettingsDialogTitle.config(text='win_length')
        if str(Event.widget) == ".!frame15.!button4": #hop_length
            AcousticSettingsDialog.config(text="Number of samples to advance between windows")
            AcousticsSettingsDialogTitle.config(text='hop_length')
        if str(Event.widget) == ".!frame15.!button6": #n_fft
            AcousticSettingsDialog.config(text="Length of the transformed axis of the output. \n If n is smaller than " \
                                   "the length of \n the win_length, the input is cropped")
            AcousticsSettingsDialogTitle.config(text='n_fft')
        if str(Event.widget) == ".!frame15.!button8": #max_F0
            AcousticSettingsDialog.config(text="Maximum allowable fundamental frequency \n of signal in Hz")
            AcousticsSettingsDialogTitle.config(text='max_F0')
        if str(Event.widget) == ".!frame15.!button10": #min_frequency
            AcousticSettingsDialog.config(text="Lower frequency cutoff in Hz. Only power at \n frequencies above " \
                                   "this will contribute to \n feature calculation")
            AcousticsSettingsDialogTitle.config(text='min_frequency')
        if str(Event.widget) == ".!frame15.!button12": #freq_range
            AcousticSettingsDialog.config(text="Proportion of power spectrum \n frequency bins to consider")
            AcousticsSettingsDialogTitle.config(text='freq_range')
        if str(Event.widget) == ".!frame15.!button14": #baseline_amp
            AcousticSettingsDialog.config(text="Baseline amplitude used to \n calculate amplitude in dB")
            AcousticsSettingsDialogTitle.config(text='baseline_amp')
        if str(Event.widget) == ".!frame15.!button16": #fmax_yin
            AcousticSettingsDialog.config(text="Maximum frequency in Hz used to \n estimate fundamental frequency " \
                                   "with \n the YIN algorithm")
            AcousticsSettingsDialogTitle.config(text='fmax_yin')

        AcousticSettingsDialog.update()
        ##############################################################################3
        if str(Event.widget) == ".!frame8.!button2": #bandpass_lower_cutoff
            LabelingSpectrogramDialog.config(text="Lower cutoff frequency in Hz for a hamming window \n" \
                                   "bandpass filter applied to the audio data before generating\n" \
                                   "spectrograms. Frequencies below this value will be filtered out")
        if str(Event.widget) == ".!frame8.!button4": #bandpass_upper_cutoff
            LabelingSpectrogramDialog.config(text="Upper cutoff frequency in Hz for a hamming window bandpass\n" \
                                   " filter applied to the audio data before generating spectrograms. \n" \
                                   "Frequencies above this value will be filtered out")
        if str(Event.widget) == ".!frame8.!button6": #a_min
            LabelingSpectrogramDialog.config(text="Minimum amplitude threshold in the spectrogram. Values \n" \
                                   "lower than a_min will be set to a_min before conversion to decibels")
        if str(Event.widget) == ".!frame8.!button8": #ref_db
            LabelingSpectrogramDialog.config(text="When making the spectrogram and converting it from amplitude \n" \
                                   "to db, the amplitude is scaled relative to this reference: \n" \
                                   "20 * log10(S/ref_db) where S represents the spectrogram with amplitude values")
        if str(Event.widget) == ".!frame8.!button10": #min_level_db
            LabelingSpectrogramDialog.config(text="When making the spectrogram, once the amplitude has been converted \n" \
                                   "to decibels, the spectrogram is normalized according to this value: \n" \
                                   "(S - min_level_db)/-min_level_db where S represents the spectrogram \n" \
                                   "in db. Any values of the resulting operation which are <0 are set to \n" \
                                   "0 and any values that are >1 are set to 1")
        if str(Event.widget) == ".!frame8.!button12": #n_fft
            LabelingSpectrogramDialog.config(text="When making the spectrogram, this is the length of the windowed \n" \
                                   "signal after padding with zeros. The number of rows spectrogram is\n" \
                                   " \"(1+n_fft/2)\". The default value,\"n_fft=512\" samples, \n" \
                                   "corresponds to a physical duration of 93 milliseconds at a sample \n" \
                                   "rate of 22050 Hz, i.e. the default sample rate in librosa. This value \n" \
                                   "is well adapted for music signals. However, in speech processing, the \n" \
                                   "recommended value is 512, corresponding to 23 milliseconds at a sample\n" \
                                   " rate of 22050 Hz. In any case, we recommend setting \"n_fft\" to a \n" \
                                   "power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm")
        if str(Event.widget) == ".!frame8.!button14": #win_length
            LabelingSpectrogramDialog.config(text="When making the spectrogram, each frame of audio is windowed by a window \n" \
                                   "of length \"win_length\" and then padded with zeros to match \"n_fft\".\n" \
                                   " Padding is added on both the left- and the right-side of the window so\n" \
                                   " that the window is centered within the frame. Smaller values improve \n" \
                                   "the temporal resolution of the STFT (i.e. the ability to discriminate \n" \
                                   "impulses that are closely spaced in time) at the expense of frequency \n" \
                                   "resolution (i.e. the ability to discriminate pure tones that are closely\n" \
                                   " spaced in frequency). This effect is known as the time-frequency \n" \
                                   "localization trade-off and needs to be adjusted according to the \n" \
                                   "properties of the input signal")
        if str(Event.widget) == ".!frame8.!button16": #hop_length
            LabelingSpectrogramDialog.config(text="The number of audio samples between adjacent windows when creating \n" \
                                   "the spectrogram. Smaller values increase the number of columns in \n" \
                                   "the spectrogram without affecting the frequency resolution")
        if str(Event.widget) == ".!frame8.!button18": #max_spec_size
            LabelingSpectrogramDialog.config(text="Maximum frequency in Hz used to \n estimate fundamental frequency " \
                                   "with \n the YIN algorithm")
        if str(Event.widget) == ".!frame9.!button2": #n_neighbors
            LabelingUMAPDialog.config(text="The size of local neighborhood (in terms of number of neighboring sample points)\n" \
                                   " used for manifold approximation. Larger values result in more global views \n" \
                                   "of the manifold, while smaller values result in more local data being \n" \
                                   "preserved. In general values should be in the range 2 to 100")
        if str(Event.widget) == ".!frame9.!button4": #n_components
            LabelingUMAPDialog.config(text="The dimension of the space to embed into. This defaults to 2 to provide \n" \
                                   "easy visualization, but can reasonably be set to any integer value \n" \
                                   "in the range 2 to 100")
        if str(Event.widget) == ".!frame9.!button6": #min_dist
            LabelingUMAPDialog.config(text="The effective minimum distance between embedded points. \n" \
                                   "Smaller values will result in a more clustered/clumped \n" \
                                   "embedding where nearby points on the manifold are drawn \n" \
                                   "closer together, while larger values will result on a more \n" \
                                   "even dispersal of points. The value should be set relative to \n" \
                                   "the \"spread\" value, which determines the scale at which \n" \
                                   "embedded points will be spread out")
        if str(Event.widget) == ".!frame9.!button8": #spread
            LabelingUMAPDialog.config(text="The effective scale of embedded points. In combination with \n" \
                                   "\"min_dist\" this determines how clustered/clumped the embedded points are")
        if str(Event.widget) == ".!frame9.!button10": #metric
            LabelingUMAPDialog.config(text="The metric to use to compute distances in high dimensional space")
        if str(Event.widget) == ".!frame9.!button12": #random_state
            LabelingUMAPDialog.config(text="If specified, random_state is the seed used by the random \n" \
                                   "number generator. Specifying a random state is the only way \n" \
                                   "to ensure that you can reproduce an identical UMAP with the \n" \
                                   "same data set multiple times")
        if str(Event.widget) == ".!frame10.!button2":
            LabelingClusterDialog.config(text='Minimum fraction of syllables that can constitute a cluster. \n'
                                              'For example, in a dataset of 1000 syllables, there need to be \n'
                                              'at least 40 instances of a particular syllable for that to be \n'
                                              'considered a cluster. Single linkage splits that contain fewer \n'
                                              'points than this will be considered points “falling out” of a \n'
                                              'cluster rather than a cluster splitting into two new clusters \n'
                                              'when performing HDBSCAN clustering')
        if str(Event.widget) == ".!frame10.!button4":
            LabelingClusterDialog.config(text='The number of samples in a neighbourhood for a point to be \n'
                                              'considered a core point in HDBSCAN clustering. The larger the \n'
                                              'value of \"min_samples\" you provide, the more conservative \n'
                                              'the clustering – more points will be declared as noise, and \n'
                                              'clusters will be restricted to progressively more dense areas')
        if str(Event.widget) == ".!frame19.!button2":
            SyntaxDialog.config(text="Minimum duration in seconds for a gap between syllables \n"
                                     "to be considered syntactically relevant. This value should \n"
                                     "be selected such that gaps between syllables in a bout are \n"
                                     "shorter than min_gap, but gaps between bouts are longer than min_gap")

    def LessInfo(Event):
        if "frame15" in str(Event.widget):
            AcousticSettingsDialog.config(text="")
            AcousticsSettingsDialogTitle.config(text="")
        if "frame8" in str(Event.widget):
            LabelingSpectrogramDialog.config(text="")
        if "frame9" in str(Event.widget):
            LabelingUMAPDialog.config(text="")
        if "frame10" in str(Event.widget):
            LabelingClusterDialog.config(text="")


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
    LabelingNotebook.add(LabelingSettingsFrame, text="Spectrogram Settings")
    LabelingUMAPSettings = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingUMAPSettings, text="UMAP Settings")
    LabelingClusterSettings = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingClusterSettings, text="Clustering Settings")
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

    LabelingBirdIDText2 = tk.Entry(LabelingBulkSave, font=("Arial", 15), justify="center", fg="grey")
    LabelingBirdIDText2.insert(0, "Bird ID")
    LabelingBirdIDText2.bind("<FocusIn>", labeling_handle_focus_in_2)
    LabelingBirdIDText2.bind("<FocusOut>", labeling_handle_focus_out_2)
    LabelingBirdIDText2.grid(row=2, column=1, sticky="w")

    LabelingBirdIDLabel2 = tk.Label(LabelingBulkSave, text="Bird ID:")
    LabelingBirdIDLabel2.grid(row=2, column=0)
    LabelingFileExplorer2 = tk.Button(LabelingBulkSave, text="Find Labeling File",command=lambda: LabelingFileExplorerFunctionSaving())
    LabelingFileExplorer2.grid(row=0, column=0)

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

    ### Labeling Spectrogram Settings ###
    def ResetLabelingSetting(Variable, EntryList):
        DefaultValues = [500, 15000, 0.00001, 2, -28, 512, 512, 128, 300]
        if Variable == "all":
            bandpass_lower_cutoff_entry_Labeling.delete(0, END)
            bandpass_lower_cutoff_entry_Labeling.insert(0, str(DefaultValues[0]))
            bandpass_lower_cutoff_entry_Labeling.update()

            bandpass_upper_cutoff_entry_Labeling.delete(0, END)
            bandpass_upper_cutoff_entry_Labeling.insert(0, str(DefaultValues[1]))
            bandpass_upper_cutoff_entry_Labeling.update()

            a_min_entry_Labeling.delete(0, END)
            a_min_entry_Labeling.insert(0, str(DefaultValues[2]))
            a_min_entry_Labeling.update()

            ref_db_entry_Labeling.delete(0, END)
            ref_db_entry_Labeling.insert(0, str(DefaultValues[3]))
            ref_db_entry_Labeling.update()

            min_level_db_entry_Labeling.delete(0, END)
            min_level_db_entry_Labeling.insert(0, str(DefaultValues[4]))
            min_level_db_entry_Labeling.update()

            n_fft_entry_Labeling.delete(0, END)
            n_fft_entry_Labeling.insert(0, str(DefaultValues[5]))
            n_fft_entry_Labeling.update()

            win_length_entry_Labeling.delete(0, END)
            win_length_entry_Labeling.insert(0, str(DefaultValues[6]))
            win_length_entry_Labeling.update()

            hop_length_entry_Labeling.delete(0, END)
            hop_length_entry_Labeling.insert(0, str(DefaultValues[7]))
            hop_length_entry_Labeling.update()

            max_spec_size_entry_Labeling.delete(0, END)
            max_spec_size_entry_Labeling.insert(0, str(DefaultValues[8]))
            max_spec_size_entry_Labeling.update()
        else:
            Variable.delete(0, END)
            Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
            Variable.update()

    LabelingSpectrogramDialog = tk.Label(LabelingSettingsFrame, text="", justify="center")
    LabelingSpectrogramDialog.grid(row=0, column=4, rowspan=8)

    LabelingUMAPDialog = tk.Label(LabelingUMAPSettings, text="", justify="center")
    LabelingUMAPDialog.grid(row=0, column=4, rowspan=6)

    bandpass_lower_cutoff_text_Labeling = tk.Label(LabelingSettingsFrame, text="bandpass_lower_cutoff").grid(row=0, column=0)
    bandpass_lower_cutoff_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    bandpass_lower_cutoff_entry_Labeling.insert(0, "500")
    bandpass_lower_cutoff_entry_Labeling.grid(row=0, column=1)
    bandpass_lower_cutoff_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(bandpass_lower_cutoff_entry_Labeling, LabelSpecList)).grid(row=0, column=2)
    bandpass_lower_cutoff_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    bandpass_lower_cutoff_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    bandpass_lower_cutoff_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    bandpass_lower_cutoff_moreinfo_Labeling.bind("<Leave>", LessInfo)

    bandpass_upper_cutoff_text_Labeling = tk.Label(LabelingSettingsFrame, text="bandpass_upper_cutoff").grid(row=1,column=0)
    bandpass_upper_cutoff_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    bandpass_upper_cutoff_entry_Labeling.insert(0, "15000")
    bandpass_upper_cutoff_entry_Labeling.grid(row=1, column=1)
    bandpass_upper_cutoff_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(bandpass_upper_cutoff_entry_Labeling, LabelSpecList)).grid(row=1, column=2)
    bandpass_upper_cutoff_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    bandpass_upper_cutoff_moreinfo_Labeling.grid(row=1, column=3, sticky=W)
    bandpass_upper_cutoff_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    bandpass_upper_cutoff_moreinfo_Labeling.bind("<Leave>", LessInfo)

    a_min_text_Labeling = tk.Label(LabelingSettingsFrame, text="a_min").grid(row=2,column=0)
    a_min_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    a_min_entry_Labeling.insert(0, "0.00001")
    a_min_entry_Labeling.grid(row=2, column=1)
    a_min_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(a_min_entry_Labeling, LabelSpecList)).grid(row=2, column=2)
    a_min_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    a_min_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
    a_min_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    a_min_moreinfo_Labeling.bind("<Leave>", LessInfo)

    ref_db_text_Labeling = tk.Label(LabelingSettingsFrame, text="ref_db").grid(row=3,column=0)
    ref_db_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    ref_db_entry_Labeling.insert(0, "20")
    ref_db_entry_Labeling.grid(row=3, column=1)
    ref_db_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(ref_db_entry_Labeling, LabelSpecList)).grid(row=3, column=2)
    ref_db_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    ref_db_moreinfo_Labeling.grid(row=3, column=3, sticky=W)
    ref_db_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    ref_db_moreinfo_Labeling.bind("<Leave>", LessInfo)

    min_level_db_text_Labeling = tk.Label(LabelingSettingsFrame, text="min_level_db").grid(row=4,column=0)
    min_level_db_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    min_level_db_entry_Labeling.insert(0, "-28")
    min_level_db_entry_Labeling.grid(row=4, column=1)
    min_level_db_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(min_level_db_entry_Labeling, LabelSpecList)).grid(row=4, column=2)
    min_level_db_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    min_level_db_moreinfo_Labeling.grid(row=4, column=3, sticky=W)
    min_level_db_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_level_db_moreinfo_Labeling.bind("<Leave>", LessInfo)

    n_fft_text_Labeling = tk.Label(LabelingSettingsFrame, text="n_fft").grid(row=5,column=0)
    n_fft_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    n_fft_entry_Labeling.insert(0, "512")
    n_fft_entry_Labeling.grid(row=5, column=1)
    n_fft_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(n_fft_entry_Labeling, LabelSpecList)).grid(row=5, column=2)
    n_fft_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    n_fft_moreinfo_Labeling.grid(row=5, column=3, sticky=W)
    n_fft_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_fft_moreinfo_Labeling.bind("<Leave>", LessInfo)

    win_length_text_Labeling = tk.Label(LabelingSettingsFrame, text="win_length").grid(row=6,column=0)
    win_length_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    win_length_entry_Labeling.insert(0, "512")
    win_length_entry_Labeling.grid(row=6, column=1)
    win_length_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(win_length_entry_Labeling, LabelSpecList)).grid(row=6, column=2)
    win_length_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    win_length_moreinfo_Labeling.grid(row=6, column=3, sticky=W)
    win_length_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    win_length_moreinfo_Labeling.bind("<Leave>", LessInfo)

    hop_length_text_Labeling = tk.Label(LabelingSettingsFrame, text="hop_length").grid(row=7,column=0)
    hop_length_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    hop_length_entry_Labeling.insert(0, "128")
    hop_length_entry_Labeling.grid(row=7, column=1)
    hop_length_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(hop_length_entry_Labeling, LabelSpecList)).grid(row=7, column=2)
    hop_length_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    hop_length_moreinfo_Labeling.grid(row=7, column=3, sticky=W)
    hop_length_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    hop_length_moreinfo_Labeling.bind("<Leave>", LessInfo)

    max_spec_size_text_Labeling = tk.Label(LabelingSettingsFrame, text="max_spec_size").grid(row=8,column=0)
    max_spec_size_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    max_spec_size_entry_Labeling.insert(0, "300")
    max_spec_size_entry_Labeling.grid(row=8, column=1)
    max_spec_size_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(max_spec_size_entry_Labeling, LabelSpecList)).grid(row=8, column=2)
    max_spec_size_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    max_spec_size_moreinfo_Labeling.grid(row=8, column=3, sticky=W)
    max_spec_size_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    max_spec_size_moreinfo_Labeling.bind("<Leave>", LessInfo)

    LabelSpecList = [bandpass_lower_cutoff_entry_Labeling,bandpass_upper_cutoff_entry_Labeling,a_min_entry_Labeling,ref_db_entry_Labeling,
                     min_level_db_entry_Labeling,n_fft_entry_Labeling,win_length_entry_Labeling,hop_length_entry_Labeling,max_spec_size_entry_Labeling]
    LabelingSpectrogram_ResetAllSettings = tk.Button(LabelingSettingsFrame, text="Reset All", command=lambda:ResetLabelingSetting("all", LabelSpecList))
    LabelingSpectrogram_ResetAllSettings.grid(row=9, column=1)

    ### Labeling UMAP Settings ###
    def ResetLabelingUMAPSetting(Variable, EntryList):
        DefaultValues = [10, 2, 0.0, 1, "euclidean", "None"]
        if Variable == "all":
            n_neighbors_entry_Labeling.delete(0, END)
            n_neighbors_entry_Labeling.insert(0, str(DefaultValues[0]))
            n_neighbors_entry_Labeling.update()

            n_components_entry_Labeling.delete(0, END)
            n_components_entry_Labeling.insert(0, str(DefaultValues[1]))
            n_components_entry_Labeling.update()

            min_dist_entry_Labeling.delete(0, END)
            min_dist_entry_Labeling.insert(0, str(DefaultValues[2]))
            min_dist_entry_Labeling.update()

            spread_entry_Labeling.delete(0, END)
            spread_entry_Labeling.insert(0, str(DefaultValues[3]))
            spread_entry_Labeling.update()

            metric_variable.set(DefaultValues[4])
            metric_entry_Labeling.update()

            random_state_entry_Labeling.delete(0, END)
            random_state_entry_Labeling.insert(0, str(DefaultValues[5]))
            random_state_entry_Labeling.update()
        else:
            if Variable != metric_entry_Labeling:
                Variable.delete(0, END)
                Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
                Variable.update()
            else:
                metric_variable.set(DefaultValues[4])
                metric_entry_Labeling.update()

    n_neighbors_text_Labeling = tk.Label(LabelingUMAPSettings, text="n_neighbors").grid(row=0, column=0)
    n_neighbors_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    n_neighbors_entry_Labeling.insert(0, "10")
    n_neighbors_entry_Labeling.grid(row=0, column=1)
    n_neighbors_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=0, column=2)
    n_neighbors_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    n_neighbors_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    n_neighbors_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_neighbors_moreinfo_Labeling.bind("<Leave>", LessInfo)

    n_components_text_Labeling = tk.Label(LabelingUMAPSettings, text="n_components").grid(row=1, column=0)
    n_components_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    n_components_entry_Labeling.insert(0, "2")
    n_components_entry_Labeling.grid(row=1, column=1)
    n_components_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=1, column=2)
    n_components_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    n_components_moreinfo_Labeling.grid(row=1, column=3, sticky=W)
    n_components_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_components_moreinfo_Labeling.bind("<Leave>", LessInfo)

    min_dist_text_Labeling = tk.Label(LabelingUMAPSettings, text="min_dist").grid(row=2, column=0)
    min_dist_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    min_dist_entry_Labeling.insert(0, "0.0")
    min_dist_entry_Labeling.grid(row=2, column=1)
    min_dist_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=2, column=2)
    min_dist_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    min_dist_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
    min_dist_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_dist_moreinfo_Labeling.bind("<Leave>", LessInfo)

    spread_text_Labeling = tk.Label(LabelingUMAPSettings, text="spread").grid(row=3, column=0)
    spread_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    spread_entry_Labeling.insert(0, "1.0")
    spread_entry_Labeling.grid(row=3, column=1)
    spread_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=3, column=2)
    spread_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    spread_moreinfo_Labeling.grid(row=3, column=3, sticky=W)
    spread_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    spread_moreinfo_Labeling.bind("<Leave>", LessInfo)

    metric_text_Labeling = tk.Label(LabelingUMAPSettings, text="metric").grid(row=4, column=0)
    options_list = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','mahalanobis','wminkowski','seuclidean','cosine','correlation','haversine','hamming','jaccard','dice','russelrao','kulsinski','ll_dirichlet','hellinger','rogerstanimoto','sokalmichener','sokalsneath','yule']
    metric_variable = StringVar()
    metric_variable.set(options_list[0])
    metric_entry_Labeling = tk.OptionMenu(LabelingUMAPSettings, metric_variable, *options_list)
    metric_entry_Labeling.grid(row=4, column=1)
    metric_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=4, column=2)
    metric_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    metric_moreinfo_Labeling.grid(row=4, column=3, sticky=W)
    metric_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    metric_moreinfo_Labeling.bind("<Leave>", LessInfo)

    random_state_text_Labeling = tk.Label(LabelingUMAPSettings, text="random_state").grid(row=5, column=0)
    random_state_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    random_state_entry_Labeling.insert(0, "None")
    random_state_entry_Labeling.grid(row=5, column=1)
    random_state_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars)).grid(row=5, column=2)
    random_state_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    random_state_moreinfo_Labeling.grid(row=5, column=3, sticky=W)
    random_state_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    random_state_moreinfo_Labeling.bind("<Leave>", LessInfo)


    LabelingUMAPVars = [n_neighbors_entry_Labeling,n_components_entry_Labeling,min_dist_entry_Labeling,spread_entry_Labeling,metric_entry_Labeling,random_state_entry_Labeling]
    LabelingUMAPSettingsResetAll = tk.Button(LabelingUMAPSettings, text="Reset All", command=lambda:ResetLabelingUMAPSetting("all", LabelingUMAPVars))
    LabelingUMAPSettingsResetAll.grid(row=6, column=1)

    ### Labeling Clustering Settings ###
    def ResetLabelingClusterSetting(Variable, EntryList):
        DefaultValues = [0.04, 5]
        if Variable == "all":
            min_cluster_prop_entry_Labeling.delete(0, END)
            min_cluster_prop_entry_Labeling.insert(0, str(DefaultValues[0]))
            min_cluster_prop_entry_Labeling.update()

            spread_entry_Labeling.delete(0, END)
            spread_entry_Labeling.insert(0, str(DefaultValues[1]))
            spread_entry_Labeling.update()
        else:
            Variable.delete(0, END)
            Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
            Variable.update()

    min_cluster_prop_text_Labeling = tk.Label(LabelingClusterSettings, text="spread").grid(row=0, column=0)
    min_cluster_prop_entry_Labeling = tk.Entry(LabelingClusterSettings, justify="center")
    min_cluster_prop_entry_Labeling.insert(0, "0.04")
    min_cluster_prop_entry_Labeling.grid(row=0, column=1)
    min_cluster_prop_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(min_cluster_prop_entry_Labeling,LabelingClusterVars)).grid(row=0, column=2)
    min_cluster_prop_moreinfo_Labeling = tk.Button(LabelingClusterSettings, text='?', state="disabled", fg="black")
    min_cluster_prop_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    min_cluster_prop_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_cluster_prop_moreinfo_Labeling.bind("<Leave>", LessInfo)

    spread_text_Labeling = tk.Label(LabelingClusterSettings, text="spread").grid(row=1, column=0)
    spread_entry_Labeling = tk.Entry(LabelingClusterSettings, justify="center")
    spread_entry_Labeling.insert(0, "5")
    spread_entry_Labeling.grid(row=1, column=1)
    spread_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(n_neighbors_entry_Labeling,LabelingClusterVars)).grid(row=1, column=2)
    spread_moreinfo_Labeling = tk.Button(LabelingClusterSettings, text='?', state="disabled", fg="black")
    spread_moreinfo_Labeling.grid(row=1, column=3, sticky=W)
    spread_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    spread_moreinfo_Labeling.bind("<Leave>", LessInfo)

    LabelingClusterVars = [min_cluster_prop_entry_Labeling,spread_entry_Labeling]
    LabelingClusterResetAll = tk.Button(LabelingClusterSettings, text="Reset All", command=lambda: ResetLabelingClusterSetting("all", LabelingClusterVars))
    LabelingClusterResetAll.grid(row=2, column=1)

    LabelingClusterDialog = tk.Label(LabelingClusterSettings, text="")
    LabelingClusterDialog.grid(row=0, column=4, rowspan=3)

    def LoadSettings_Labeling():
        SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        if SettingsMetadata != "":
            SettingsMetadata_df = pd.read_csv(SettingsMetadata)
            bandpass_lower_cutoff_entry_Labeling.delete(0,END)
            bandpass_lower_cutoff_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][0]))

            bandpass_upper_cutoff_entry_Labeling.delete(0, END)
            bandpass_upper_cutoff_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][1]))

            a_min_entry_Labeling.delete(0, END)
            a_min_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][2]))

            ref_db_entry_Labeling.delete(0, END)
            ref_db_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][3]))

            min_level_db_entry_Labeling.delete(0, END)
            min_level_db_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][4]))

            n_fft_entry_Labeling.delete(0, END)
            n_fft_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][5]))

            win_length_entry_Labeling.delete(0, END)
            win_length_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][6]))

            hop_length_entry_Labeling.delete(0, END)
            hop_length_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][7]))

            max_spec_size_entry_Labeling.delete(0, END)
            max_spec_size_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][8]))
            ###########################################
            n_neighbors_entry_Labeling.delete(0, END)
            n_neighbors_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP Value"][0]))

            n_components_entry_Labeling.delete(0, END)
            n_components_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP Value"][1]))

            min_dist_entry_Labeling.delete(0, END)
            min_dist_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP Value"][2]))

            spread_entry_Labeling.delete(0, END)
            spread_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP Value"][3]))

            metric_variable.set(str(SettingsMetadata_df["UMAP Value"][4]))

            random_state_entry_Labeling.delete(0, END)
            if str(SettingsMetadata_df["UMAP Value"][5]) == "nan":
                random_state_entry_Labeling.insert(0, "None")
            else:
                random_state_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP Value"][5]))
            ###################
            min_cluster_prop_entry_Labeling.delete(0, END)
            min_cluster_prop_entry_Labeling.insert(0, str(SettingsMetadata_df["Clustering Values"][0]))

            spread_entry_Labeling.delete(0, END)
            spread_entry_Labeling.insert(0, str(SettingsMetadata_df["Clustering Values"][1]))

    LoadLabelingSettings = tk.Button(LabelingSettingsFrame, text="Load Settings", command=lambda: LoadSettings_Labeling())
    LoadLabelingSettings.grid(row=9, column=0)

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

    AcousticsOnsetLabel = tk.Label(AcousticsMainFrameSingle, text="Onset:").grid(row=3, column=0)
    AcousticsOnset = tk.Entry(AcousticsMainFrameSingle, justify="center")
    AcousticsOnset.insert(0,"0")
    AcousticsOnset.grid(row=4, column=0)

    AcousticsOffsetLabel = tk.Label(AcousticsMainFrameSingle, text="Offset:").grid(row=5, column=0)
    AcousticsOffset = tk.Entry(AcousticsMainFrameSingle, justify="center")
    AcousticsOffset.insert(0, "End of File")
    AcousticsOffset.grid(row=6, column=0)

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
    AcousticsInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsInfoFrame, text="Info")

    ### Advanced Acoustics Settings ###

    AcousticsSettingsDialogTitle = tk.Label(AcousticsSettingsFrame, text="", justify="center", font=("Arial", 25, "bold"))
    AcousticsSettingsDialogTitle.grid(row=0, column=4)
    AcousticSettingsDialog = tk.Label(AcousticsSettingsFrame, text="", justify="center")
    AcousticSettingsDialog.grid(row=1, column=4, rowspan=8)

    def ResetAcousticSetting(Variable, EntryList):
        DefaultValues = [400,40,1024,1830,380,0.5,70, 8000]
        if Variable == "all":
            win_length_entry_Acoustics.delete(0, END)
            win_length_entry_Acoustics.insert(0, str(DefaultValues[0]))
            win_length_entry_Acoustics.update()

            hop_length_entry_Acoustics.delete(0, END)
            hop_length_entry_Acoustics.insert(0, str(DefaultValues[1]))
            hop_length_entry_Acoustics.update()

            n_fft_entry_Acoustics.delete(0, END)
            n_fft_entry_Acoustics.insert(0, str(DefaultValues[2]))
            n_fft_entry_Acoustics.update()

            max_F0_entry_Acoustics.delete(0, END)
            max_F0_entry_Acoustics.insert(0, str(DefaultValues[3]))
            max_F0_entry_Acoustics.update()

            min_frequency_entry_Acoustics.delete(0, END)
            min_frequency_entry_Acoustics.insert(0, str(DefaultValues[4]))
            min_frequency_entry_Acoustics.update()

            freq_range_entry_Acoustics.delete(0, END)
            freq_range_entry_Acoustics.insert(0, str(DefaultValues[5]))
            freq_range_entry_Acoustics.update()

            baseline_amp_entry_Acoustics.delete(0, END)
            baseline_amp_entry_Acoustics.insert(0, str(DefaultValues[6]))
            baseline_amp_entry_Acoustics.update()

            fmax_yin_entry_Acoustics.delete(0, END)
            fmax_yin_entry_Acoustics.insert(0, str(DefaultValues[7]))
            fmax_yin_entry_Acoustics.update()


        else:
            Variable.delete(0, END)
            Variable.insert(0,str(DefaultValues[EntryList.index(Variable)]))
            Variable.update()

    win_length_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="win_length").grid(row=0, column=0)
    win_length_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    win_length_entry_Acoustics.insert(0, "400")
    win_length_entry_Acoustics.grid(row=0, column=1)
    win_length_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(win_length_entry_Acoustics, EntryList)).grid(row=0, column=2)
    win_length_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    win_length_moreinfo_Acoustics.grid(row=0, column=3, sticky=W)
    win_length_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    win_length_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    hop_length_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="hop_length").grid(row=1, column=0)
    hop_length_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    hop_length_entry_Acoustics.insert(0, "40")
    hop_length_entry_Acoustics.grid(row=1, column=1)
    hop_length_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(hop_length_entry_Acoustics, EntryList)).grid(row=1, column=2)
    hop_length_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    hop_length_moreinfo_Acoustics.grid(row=1, column=3, sticky=W)
    hop_length_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    hop_length_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    n_fft_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="n_fft").grid(row=2, column=0)
    n_fft_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    n_fft_entry_Acoustics.insert(0, "1024")
    n_fft_entry_Acoustics.grid(row=2, column=1)
    n_fft_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(n_fft_entry_Acoustics, EntryList)).grid(row=2, column=2)
    n_fft_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    n_fft_moreinfo_Acoustics.grid(row=2, column=3, sticky=W)
    n_fft_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    n_fft_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    max_F0_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="max_F0").grid(row=3, column=0)
    max_F0_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    max_F0_entry_Acoustics.insert(0, "1830")
    max_F0_entry_Acoustics.grid(row=3, column=1)
    max_F0_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(max_F0_entry_Acoustics, EntryList)).grid(row=3, column=2)
    max_F0_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    max_F0_moreinfo_Acoustics.grid(row=3, column=3, sticky=W)
    max_F0_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    max_F0_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    min_frequency_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="min_frequency").grid(row=4, column=0)
    min_frequency_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    min_frequency_entry_Acoustics.insert(0, "380")
    min_frequency_entry_Acoustics.grid(row=4, column=1)
    min_frequency_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(min_frequency_entry_Acoustics, EntryList)).grid(row=4, column=2)
    min_frequency_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    min_frequency_moreinfo_Acoustics.grid(row=4, column=3, sticky=W)
    min_frequency_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    min_frequency_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    freq_range_text_Acoustics=tk.Label(AcousticsSettingsFrame, text="freq_range").grid(row=5, column=0)
    freq_range_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    freq_range_entry_Acoustics.insert(0, "0.5")
    freq_range_entry_Acoustics.grid(row=5, column=1)
    freq_range_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(freq_range_entry_Acoustics, EntryList)).grid(row=5, column=2)
    freq_range_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    freq_range_moreinfo_Acoustics.grid(row=5, column=3, sticky=W)
    freq_range_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    freq_range_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    baseline_amp_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="baseline_amp").grid(row=6, column=0)
    baseline_amp_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    baseline_amp_entry_Acoustics.insert(0, "70")
    baseline_amp_entry_Acoustics.grid(row=6, column=1)
    baseline_amp_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(baseline_amp_entry_Acoustics, EntryList)).grid(row=6, column=2)
    baseline_amp_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    baseline_amp_moreinfo_Acoustics.grid(row=6, column=3, sticky=W)
    baseline_amp_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    baseline_amp_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    fmax_yin_text_Acoustics=tk.Label(AcousticsSettingsFrame, text="fmax_yin").grid(row=7, column=0)
    fmax_yin_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    fmax_yin_entry_Acoustics.insert(0, "8000")
    fmax_yin_entry_Acoustics.grid(row=7, column=1)
    fmax_yin_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(fmax_yin_entry_Acoustics, EntryList)).grid(row=7, column=2)
    fmax_yin_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    fmax_yin_moreinfo_Acoustics.grid(row=7, column=3, sticky=W)
    fmax_yin_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    fmax_yin_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    EntryList = [win_length_entry_Acoustics, hop_length_entry_Acoustics, n_fft_entry_Acoustics, max_F0_entry_Acoustics,
                 min_frequency_entry_Acoustics, freq_range_entry_Acoustics, baseline_amp_entry_Acoustics,
                 fmax_yin_entry_Acoustics]

    AcousticResetAllVariables = tk.Button(AcousticsSettingsFrame, text="Reset All", command=lambda:ResetAcousticSetting("all", EntryList))
    AcousticResetAllVariables.grid(row=8, column=1)

    MultiAcousticsFileExplorer = tk.Button(AcousticsMainFrameMulti, text="Find Segmentation File",command=lambda: MultiAcousticsFileExplorerFunction())
    MultiAcousticsFileExplorer.grid(row=1, column=0)
    MultiAcousticsText = tk.Label(AcousticsMainFrameMulti, text="Please select which acoustics features you would like to analyze:")
    MultiAcousticsText.grid(row=0, column=0, columnspan=3)

    def LoadSettings_Acoustic():
        SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        if SettingsMetadata != "":
            SettingsMetadata_df = pd.read_csv(SettingsMetadata)
            win_length_entry_Acoustics.delete(0,END)
            win_length_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][0]))

            hop_length_entry_Acoustics.delete(0, END)
            hop_length_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][1]))

            n_fft_entry_Acoustics.delete(0, END)
            n_fft_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][2]))

            max_F0_entry_Acoustics.delete(0, END)
            max_F0_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][3]))

            min_frequency_entry_Acoustics.delete(0, END)
            min_frequency_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][4]))

            freq_range_entry_Acoustics.delete(0, END)
            freq_range_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][5]))

            baseline_amp_entry_Acoustics.delete(0, END)
            baseline_amp_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][6]))

            fmax_yin_entry_Acoustics.delete(0, END)
            fmax_yin_entry_Acoustics.insert(0, str(SettingsMetadata_df["Value"][7]))

    LoadLabelingSettings = tk.Button(AcousticsSettingsFrame, text="Load Settings", command=lambda: LoadSettings_Acoustic())
    LoadLabelingSettings.grid(row=9, column=0)

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

    ### Syntax ###
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

    def ResetSyntaxSetting():
        min_gap_entry_Syntax.delete(0, END)
        min_gap_entry_Syntax.insert(0, "0.2")
        min_gap_entry_Syntax.update()

    min_gap_text_Syntax = tk.Label(SyntaxSettingsFrame, text="min_gap").grid(row=0, column=0)
    min_gap_entry_Syntax = tk.Entry(SyntaxSettingsFrame, justify="center")
    min_gap_entry_Syntax.insert(0, "0.2")
    min_gap_entry_Syntax.grid(row=0, column=1)
    min_gap_reset_Syntax = tk.Button(SyntaxSettingsFrame, text="Reset",command=lambda: ResetSyntaxSetting()).grid(row=0, column=2)
    min_gap_moreinfo_Syntax = tk.Button(SyntaxSettingsFrame, text='?', state="disabled", fg="black")
    min_gap_moreinfo_Syntax.grid(row=0, column=3, sticky=W)
    min_gap_moreinfo_Syntax.bind("<Enter>", MoreInfo)
    min_gap_moreinfo_Syntax.bind("<Leave>", LessInfo)

    SyntaxDialog = tk.Label(SyntaxSettingsFrame, text="", justify="center")
    SyntaxDialog.grid(row=0, column=4, rowspan=2)


    Var_Heatmap = IntVar()
    SelectCountHeatmap = tk.Radiobutton(SyntaxMainFrame, text="Count Matrix", variable=Var_Heatmap, value=1)
    SelectCountHeatmap.grid(row=3, column=0)
    SelectProbHeatmap = tk.Radiobutton(SyntaxMainFrame, text="Probability Matrix", variable=Var_Heatmap, value=2)
    SelectProbHeatmap.grid(row=3, column=1)


    def GenerateMatrixHeatmap():
        global syntax_data
        global Bird_ID
        try:
            # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
            sns.heatmap(syntax_data.trans_mat, annot=True, fmt='0.0f')
        except:
            SyllableSyntax()
            # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
            sns.heatmap(syntax_data.trans_mat, annot=True, fmt='0.0f')
        plt.title("Count Transition Matrix")
        plt.xticks(rotation=0)
        plt.savefig("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap.png")

        Heatmap = PhotoImage(file="C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap.png")
        MatrixHeatmap_Display = tk.Toplevel(gui)
        MatrixHeatmap_Display_Label = tk.Label(MatrixHeatmap_Display, image= Heatmap)
        MatrixHeatmap_Display_Label.grid(row=1, column=0, columnspan=3)
        MatrixHeatmap_SaveButton = tk.Button(MatrixHeatmap_Display, text="Save", command=lambda:plt.savefig("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/"+str(Bird_ID)+"_MatrixHeatmap.png"))
        MatrixHeatmap_SaveButton.grid(row=0, column=1)

        MatrixHeatmap_Display.mainloop()

    MatrixHeatmap_Button = tk.Button(SyntaxMainFrame, text="Generate Matrix Heatmap", command=lambda:GenerateMatrixHeatmap())
    MatrixHeatmap_Button.grid(row=10, column=0)


    """
            Changes:

            Matrix Heatmap (have option for custom title - default is "TransitionMatrix_[BirdID]")
               plt.figure(figsize = (x, y))
               sns.heatmap(syntax_data.trans_mat, annot = True, fmt = '0.0f')
               MatrixHeatmap = plt.getFigure()

            Probability Matrix - Same as above; maybe have an option to select b/w count vs probability


            Syntax Raster:
            Option to choose alignment syllable (or have no alignment syllable to not align anything)
            Create preview for syntax raster w/o alignment to determine best syllable to use

            Make syntax_data object global to use for transition matrix and syntax raster; have matrix and raster 
            look for syntax_data to see if it exists, otherwise run syntax to generate syntax_data

            Save entropy rate and entropy rate normalized to a csv

            Give option to save syllable repetitions to csv

            For all output things (for all modules) create "Output" folder containing subfolders for each module's output csv's and photos

            Merged syllable stats

            Ignore syllable pair repetition

            Ignore syntax for many birds

            make error messages for every entry field

            Easter eggs?

            When saving csv's, save them to subfolder w/in module folder (folder is named as timestamp of file creation)
            """

    ### Plain Spectrogram Generation - Whole File ###
    PlainSpectro = tk.Frame(gui)
    notebook.add(PlainSpectro, text="Plain Spectrograms")
    
    PlainMainFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook = ttk.Notebook(PlainSpectro)
    PlainNotebook.grid(row=1)
    PlainNotebook.add(PlainMainFrame, text="Whole Folder")

    
    PlainFileExplorer = tk.Button(PlainMainFrame, text="Select Folder", command=lambda: FindPlainSpectrogramsFolder())
    PlainFileExplorer.grid(row=1, column=0)
    PlainSpectroRun = tk.Button(PlainMainFrame, text="Create Blank Spectrograms", command=lambda:PrintPlainSpectrograms())
    PlainSpectroRun.grid(row=2, column=0, columnspan=2)

    ### Plain Spectrogram Generation - Selected Files ###
    PlainSpectroAlt= tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainSpectroAlt, text="Selected Files")
    PlainFileExplorerAlt = tk.Button(PlainSpectroAlt, text="Select Files", command=lambda: FindPlainSpectrogramsFiles())
    PlainFileExplorerAlt.grid(row=1, column=0)
    PlainSpectroRunAlt = tk.Button(PlainSpectroAlt, text="Create Blank Spectrograms",
                                command=lambda: PrintPlainSpectrogramsAlt())
    PlainSpectroRunAlt.grid(row=2, column=0, columnspan=2)
    PlainSettingsFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainSettingsFrame, text="Advanced Settings")
    PlainInfoFrame = tk.Frame(width =MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainInfoFrame, text="Info")


    ttk.Style().theme_use("clam")
    gui.mainloop()

except Exception:
    print(Exception)