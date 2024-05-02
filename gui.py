######  Required libraries to install: ########
# avn
# pymde
# hdbscan
# Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/


#Upgrade seaborn from 0.11.2 to 0.12.2
# pandas is version 2.0.3
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

 -
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
    import avn.syntax as syntax
    import avn.timing as timing
    import avn.plotting
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
    #import customtkinter ### Causes an issue for Windows 11 on Dan's computer
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import pymde
    import matplotlib
    import matplotlib.pyplot as plt

    import hdbscan
    import math
    import sklearn
    import seaborn as sns
    from matplotlib.figure import Figure
    import audioread
    # import simple_webbrowser as swb
    import webbrowser as wb
    from tkinter.filedialog import asksaveasfile
    import shutil


    def focus_in(Event):
        global LabelingBirdIDText
        global BirdID_Labeling
        global LabelingBirdIDText2
        global BirdIDText
        global SyntaxBirdID
        global MultiAcousticsBirdID

        Entry = Event.widget
        if Entry.get() == "Bird ID":
            Entry.delete(0, tk.END)
            Entry.config(fg='black')

    def focus_out(Event):
        global LabelingBirdIDText
        global BirdID_Labeling
        global LabelingBirdIDText2
        global BirdIDText
        global SyntaxBirdID
        global MultiAcousticsBirdID

        Entry = Event.widget
        if Entry.get() == "":
            Entry.delete(0, tk.END)
            Entry.config(fg='grey')
            Entry.insert(0, "Bird ID")

    def LabelingCountFocusIn(_):
        global tempLabelingPhotoCount
        tempLabelingPhotoCount = LabelingPhotoCount.get()
        if tempLabelingPhotoCount == "0":
            LabelingPhotoCount.delete(0, tk.END)

    def LabelingCountFocusOut(_):
        global tempLabelingPhotoCount
        if tempLabelingPhotoCount == "0" and LabelingPhotoCount.get() == "":
            LabelingPhotoCount.insert(0,"0")

    def FileExplorer(Module, Type):
        # This will be used for Labeling, Generate Spectrogram/UMAP, and Timing
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')

        if Module == "Labeling":
            if Type == "Input":
                global LabelingBirdIDText
                global LabelingDirectoryText
                global segmentations_path
                segmentations_path_temp = filedialog.askopenfilenames(filetypes=[(".csv files", "*.csv")])
                if len(segmentations_path_temp) > 0:
                    if len(segmentations_path_temp) == 1:
                        segmentations_path = segmentations_path_temp
                        LabelingDirectoryText.config(text=segmentations_path)
                        LabelingDirectoryText.update()
                        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                        if pattern.search(segmentations_path[0]) != None:
                            LabelingBirdIDText.delete(0, tk.END)
                            LabelingBirdIDText.config(fg='black')
                            LabelingBirdIDText.insert(0, str(pattern.search(segmentations_path[0]).group()))
                            LabelingBirdIDText.update()
                            #Bird_ID = str(pattern.search(segmentations_path).group())
                    if len(segmentations_path_temp) > 1:
                        segmentations_path = segmentations_path_temp
                        LabelingDirectoryText.config(text=str(len(segmentations_path))+" files selected")
                        LabelingDirectoryText.update()
                        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                        if pattern.search(segmentations_path[0]) != None:
                            LabelingBirdIDText.delete(0, tk.END)
                            LabelingBirdIDText.config(fg='black')
                            LabelingBirdIDText.insert(0, str(pattern.search(segmentations_path[0]).group()))
                            LabelingBirdIDText.update()
            if Type == "Output":
                global LabelingOutputFile_Text
                Selected_Output = filedialog.askdirectory()
                if len(Selected_Output) > 0:
                    LabelingOutputFile_Text.config(text=str(Selected_Output))
                    LabelingOutputFile_Text.update()
        if Module == "Labeling_UMAP":
            if Type == "Input":
                global LabelingSpectrogramFiles
                global LabelingFileDisplay
                LabelingFile = filedialog.askopenfilenames(filetypes=[(".csv files", "*.csv")])
                if len(LabelingFile) > 0:
                    print(LabelingFile)
                    print(type(LabelingFile))
                    print(LabelingFile[0])
                    LabelingSpectrogramFiles = LabelingFile
                    LabelingFileDisplay.config(text=LabelingFile)
                    LabelingFileDisplay.update()
                    # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                    if pattern.search(LabelingFile[0]) != None:
                        global BirdID_Labeling
                        BirdID_Labeling.delete(0, tk.END)
                        BirdID_Labeling.config(fg='black')
                        BirdID_Labeling.insert(0, str(pattern.search(LabelingFile).group()))
                        BirdID_Labeling.update()
            if Type == "Output":
                global LabelingUMAP_OutputText
                OutputDir = filedialog.askdirectory()
                if len(OutputDir) > 0:
                    LabelingUMAP_OutputText.config(text=OutputDir)
                    LabelingUMAP_OutputText.update()
        if Module == "Labeling_Bulk":
            if Type == "Input":
                global labeling_path
                global Bird_ID
                global LabelingFileDisplay2
                labeling_path = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
                if len(labeling_path) > 0:
                    LabelingFileDisplay2 = tk.Label(LabelingBulkSave, text=labeling_path)
                    LabelingFileDisplay2.grid(row=0, column=1, sticky="w")

                # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                if pattern.search(labeling_path) != None:
                    LabelingBirdIDText2.delete(0, tk.END)
                    LabelingBirdIDText2.config(fg='black')
                    LabelingBirdIDText2.insert(0, str(pattern.search(labeling_path).group()))
            if Type == "Output":
                pass
        if Module == "Timing":
            if Type == "Input":
                TimingInput = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
                if len(TimingInput) > 0:
                    global TimingInput_Text
                    TimingInput_Text.config(text=TimingInput)
                    TimingInput_Text.update()
                    if pattern.search(TimingInput) != None:
                        Timing_BirdID.delete(0, tk.END)
                        Timing_BirdID.config(fg='black')
                        Timing_BirdID.insert(0, str(pattern.search(TimingInput).group()))
                        Timing_BirdID.update()
            if Type == "Output":
                TimingOutput = filedialog.askdirectory()
                if len(TimingOutput) > 0:
                    global TimingOutput_Text
                    TimingOutput_Text.config(text=TimingOutput)
                    TimingOutput_Text.update()
        if Module == "Acoustics_Single":
            if Type == "Input":
                global AcousticsInputVar
                global AcousticsDirectory
                if AcousticsInputVar.get() == 1:
                    AcousticsHarshidaInput_temp = filedialog.askdirectory()
                    if len(AcousticsHarshidaInput_temp) > 0:
                        global AcousticsHarshidaInput
                        AcousticsHarshidaInput = AcousticsHarshidaInput_temp
                        AcousticsFileDisplay.config(text=AcousticsHarshidaInput)
                if AcousticsInputVar.get() == 0:
                    AcousticsDirectory_temp = filedialog.askopenfilename(filetypes=[(".wav files", "*.wav")])
                    if len(AcousticsDirectory_temp) > 0:
                        AcousticsDirectory = AcousticsDirectory_temp
                        AcousticsFileDisplay.config(text=AcousticsDirectory)
                        AcousticsFileDisplay.update()
                        global AcousticsFileDuration
                        AcousticsFileDuration = tk.Label(AcousticsMainFrameSingle, text="          Duration: " + str(
                            audioread.audio_open(AcousticsDirectory).duration) + " sec", justify="center")
                        AcousticsFileDuration.grid(row=0, column=3)
                        ResetAcousticsOffset()
                        # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                        if pattern.search(AcousticsDirectory.split("/")[-2]) != None:
                            pass
                            # BirdIDText.delete(0, tk.END)
                            # BirdIDText.config(fg='black')
                            # BirdIDText.insert(0, str(pattern.search(AcousticsDirectory.split("/")[-2]).group()))
            if Type == "Output":
                global AcousticsOutput_Text
                OutputDir = filedialog.askdirectory()
                if len(OutputDir) > 0:
                    AcousticsOutput_Text.config(text=OutputDir)
                    AcousticsOutput_Text.update()
        if Module == "Acoustics_Multi":
            if Type == "Input":
                global MultiAcousticsDirectory
                MultiAcousticsDirectory_temp = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
                if len(MultiAcousticsDirectory_temp) > 0:
                    global MultiAcousticsInputDir
                    global MultiAcousticsBirdID
                    MultiAcousticsDirectory = MultiAcousticsDirectory_temp
                    MultiAcousticsInputDir.config(text=MultiAcousticsDirectory_temp)
                    MultiAcousticsInputDir.update()
                    # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                    if pattern.search(MultiAcousticsDirectory.split("/")[-2]) != None:
                        MultiAcousticsBirdID.delete(0, tk.END)
                        MultiAcousticsBirdID.config(fg='black')
                        MultiAcousticsBirdID.insert(0, str(pattern.search(MultiAcousticsDirectory.split("/")[-2]).group()))
            if Type == "Output":
                global MultiAcousticsOutputDisplay
                OutputDir = filedialog.askdirectory()
                if len(OutputDir) > 0:
                    MultiAcousticsOutputDisplay.config(text=OutputDir)
                    MultiAcousticsOutputDisplay.update()
        if Module == "Syntax":
            if Type == "Input":
                global SyntaxDirectory
                global AlignmentChoices
                global SyntaxAlignmentVar
                global SyntaxAlignment
                SyntaxDirectory_temp = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
                if len(SyntaxDirectory_temp) > 0:
                    global SyntaxFileDisplay
                    SyntaxFileDisplay.config(text=SyntaxDirectory_temp)
                    SyntaxFileDisplay.update()

                    if pattern.search(SyntaxDirectory_temp.split("/")[-1]) != None:
                        SyntaxBirdID.delete(0, tk.END)
                        SyntaxBirdID.config(fg='black')
                        SyntaxBirdID.insert(0, str(pattern.search(SyntaxDirectory_temp.split("/")[-1]).group()))

                    SyntaxDirectory = SyntaxDirectory_temp
                    labeling_data = pd.read_csv(SyntaxDirectory)
                    for i in labeling_data["labels"].unique():
                        AlignmentChoices.append(str(i))
                    AlignmentChoices.pop(0)
                    AlignmentChoices.sort()
                    AlignmentChoices.insert(0, "Auto")
                    SyntaxAlignmentVar.set("Auto")
                    SyntaxAlignment.destroy()
                    SyntaxAlignment = tk.OptionMenu(SyntaxMainFrame, SyntaxAlignmentVar, *AlignmentChoices)
                    SyntaxAlignment.grid(row=8, column=0)
            if Type == "Output":
                global SyntaxOutputDisplay
                OutputDir = filedialog.askdirectory()
                if len(OutputDir) > 0:
                    SyntaxOutputDisplay.config(text=OutputDir)
                    SyntaxOutputDisplay.update()
        if Module == "Plain_Folder":
            if Type == "Input":
                global PlainOutputDir
                global PlainOutputFolder_Label
                global PlainDirectoryLabel
                PlainDirectory_temp = filedialog.askdirectory()
                if len(PlainDirectory_temp) > 0:
                    PlainDirectoryLabel.config(text=PlainDirectory_temp)
                    PlainDirectoryLabel.update()
            if Type == "Output":
                global PlainOutputFolder_Label
                PlainOutputDir_temp = filedialog.askdirectory()
                if len(PlainOutputDir_temp) > 0:
                    PlainOutputFolder_Label.config(text=PlainOutputDir)
                    PlainOutputFolder_Label.update()
        if Module == "Plain_Files":
            if Type == "Input":
                global PlainDirectoryAlt
                PlainDirectoryAlt_temp = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
                if len(PlainDirectoryAlt_temp) > 0:
                    if len(PlainDirectoryAlt_temp) == 1:
                        PlainDirectoryLabelAlt.config(text=str(len(PlainDirectoryAlt_temp)) + " file selected")
                    else:
                        PlainDirectoryLabelAlt.config(text=str(len(PlainDirectoryAlt_temp)) + " files selected")
                    PlainDirectoryLabelAlt.update()
            if Type == "Output":
                global PlainOutputAlt_Label
                PlainOutputAlt_temp = filedialog.askdirectory()
                if len(PlainOutputAlt_temp) > 0:
                    PlainOutputAlt_Label.config(text=PlainOutputAlt_temp)
                    PlainOutputAlt_Label.update()
        if Module == "Labeling_Extra":
            if Type == "Input":
                global LabelingExtra_Input_Text
                global LabelingExtra_BirdID
                InputDir_temp = filedialog.askdirectory()
                if len(InputDir_temp) > 0:
                    LabelingExtra_Input_Text.config(text=InputDir_temp)
                    LabelingExtra_Input_Text.update()
                    # if pattern.search(InputDir_temp) != None:
                    #     LabelingExtra_BirdID.delete(0, tk.END)
                    #     LabelingExtra_BirdID.config(fg='black')
                    #     LabelingExtra_BirdID.insert(0, str(pattern.search(InputDir_temp).group()))
                    #     LabelingExtra_BirdID.update()
            if Type == "Output":
                global LabelingExtra_Output_Text
                OutputDir_temp = filedialog.askdirectory()
                if len(OutputDir_temp) > 0:
                    LabelingExtra_Output_Text.config(text=OutputDir_temp)
                    LabelingExtra_Output_Text.update()

    def ResetAcousticsOffset():
        AcousticsOffset.delete(0,END)
        AcousticsOffset.insert(0, audioread.audio_open(AcousticsDirectory).duration)
        AcousticsOffset.update()

    def Labeling():
        global segmentations_path
        global Bird_ID
        global song_folder
        global LabelingFileDisplay
        global LabelingErrorMessage
        global LabelingBirdIDText
        global UMAP_Directories
        global OutputFolder
        UMAP_Directories = []
        try:
            BirdID_pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
            if BirdID_pattern.search(Bird_ID) == None:
                Bird_ID = LabelingBirdIDText.cget("text")
        except:
            Bird_ID = LabelingBirdIDText.cget("text")

        # segmentations_path = LabelingDirectoryText.cget("text")

        #############################################
        seg_path_check = re.compile('(?i)[A-Z]')
        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')

        if LabelingBirdIDText.get() == "Bird ID" or LabelingBirdIDText.get() == "":
            try: # Only Bird ID missing
                LabelingFileDisplay.cget("text")
                #if "wseg" not in LabelingFileDisplay.cget("text"):
                if seg_path_check.search(segmentations_path[0]) == None:
                    LabelingErrorMessage.config(text="Invalid segmentation file and missing bird id")
                else:
                    LabelingErrorMessage.config(text="Missing bird id")
            except: # Neither file path nor Bird ID chosen
                LabelingErrorMessage.config(text="Missing segmentation folder and bird id")
        elif LabelingBirdIDText.get() != "Bird ID" and pattern.search(LabelingBirdIDText.get()) == None:
            #if "wseg" not in LabelingFileDisplay.cget("text"):
            if seg_path_check.search(segmentations_path[0]) == None:
                LabelingErrorMessage.config(text="Invalid segmentation file and invalid bird id")
            else:
                LabelingErrorMessage.config(text="Invalid Bird ID")
        elif pattern.search(LabelingBirdIDText.get()) != None:
            try:  # Only segmentation path missing
                LabelingFileDisplay.cget("text")
            except:
                LabelingErrorMessage.config(text="Missing segmentation folder")
            LabelingErrorMessage.update()
            if seg_path_check.search(segmentations_path[0]) != None:
                if seg_path_check.search(LabelingOutputFile_Text.cget("text")) == None:
                    LabelingErrorMessage.config(text="Missing output directory")
                else:
                    #global hdbscan_df
                    global EndOfFolder

                    LabelingProgress_Text = tk.Label(LabelingMainFrame, text="", justify="center")
                    LabelingProgress_Text.grid(row=9, column=0, columnspan=2)
                    gui.update_idletasks()
                    OutputFolder = LabelingOutputFile_Text.cget("text")

                    for directory in segmentations_path:

                        directory = directory.replace("\\", "/")

                        dir_corrected = directory.split("/")[-1].replace("_seg_", "_label_")
                        dir_corrected = dir_corrected.replace("_wseg.csv", str(directory.split("/")[-2])+"_labels.csv")
                        try:
                            os.makedirs(OutputFolder+"/Labeling")
                        except: pass
                        output_file = OutputFolder + "/Labeling/" + dir_corrected  # e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"

                        #############################################
                        # This is from labeling tutorial:
                        def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
                            spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length,
                                                       n_fft=n_fft)
                            spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

                            # normalize
                            S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

                            return S_norm

                        amin = 1e-5
                        ref_db = 20
                        min_level_db = -28  # ERROR! If a .wav file contains a sound that's too quiet, the audio.values variable for that row will be NULL, causing an error
                        win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
                        hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
                        n_fft = 512
                        K = 10
                        min_cluster_prop = 0.04
                        embedding_dim = 2

                        # load segmentations
                        predictions_reformat = pd.read_csv(directory)

                        segmentations = predictions_reformat
                        segmentations = segmentations.rename(
                            columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})

                        # sometimes this padding can create negative onset times, which will cause errors.
                        # correct this by setting any onsets <0 to 0.
                        segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

                        # Add syllable audio to dataframe
                        syllable_dfs = pd.DataFrame()
                        # for directory in segmentations.directory.unique():

                        LabelingProgress = IntVar()
                        LabelingProgress_Bar = ttk.Progressbar(LabelingMainFrame, orient="horizontal",
                                                               mode="determinate", maximum=100,
                                                               variable=LabelingProgress)
                        LabelingProgress_Bar.grid(row=10, column=0, columnspan=2)
                        gui.update_idletasks()

                        LabelingProgress_Text.config(text="Loading Song Files...")
                        LabelingProgress_Text.update()
                        for song_file in segmentations.files.unique():
                            LabelingProgress_Bar.step(100 / len(segmentations.files.unique()))
                            LabelingProgress_Bar.update()
                            gui.update_idletasks()
                            SongIndex = segmentations.index[segmentations['files'] == song_file].tolist()
                            file_path = segmentations.directory[SongIndex[0]].replace("\\", "/") + song_file

                            song = dataloading.SongFile(file_path)
                            song.bandpass_filter(int(bandpass_lower_cutoff_entry_Labeling.get()),
                                                 int(bandpass_upper_cutoff_entry_Labeling.get()))

                            syllable_df = segmentations[segmentations['files'] == song_file]
                            # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
                            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                                    for st, et in
                                                    zip(syllable_df.onsets.values, syllable_df.offsets.values)]
                            # print(syllable_df)
                            syllable_dfs = pd.concat([syllable_dfs, syllable_df])

                        LabelingProgress.set(0)
                        # Normalize the audio  --- Ethan's comment: This won't work when there's an empty array for syllable_dfs_audio_values, so I'm just going to set those to '[0]'
                        # print(syllable_dfs)

                        syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]

                        hdbscan_df = syllable_dfs

                        # compute spectrogram for each syllable
                        syllables_spec = []

                        LabelingProgress_Text.config(text="Generating Labels...")
                        LabelingProgress_Text.update()
                        for syllable in syllable_dfs.audio.values:
                            if len(syllable) > 0:
                                LabelingProgress_Bar.step(100 / len(syllable_dfs.audio.values))
                                LabelingProgress_Bar.update()
                                syllable_spec = make_spec(syllable,
                                                          hop_length=int(hop_length_entry_Labeling.get()),
                                                          win_length=int(win_length_entry_Labeling.get()),
                                                          n_fft=int(n_fft_entry_Labeling.get()),
                                                          ref_db=int(ref_db_entry_Labeling.get()),
                                                          amin=float(a_min_entry_Labeling.get()),
                                                          min_level_db=int(min_level_db_entry_Labeling.get()))
                                if syllable_spec.shape[1] > int(max_spec_size_entry_Labeling.get()):
                                    print(
                                        "Long Syllable Corrections! Spectrogram Duration = " + str(
                                            syllable_spec.shape[1]))
                                    syllable_spec = syllable_spec[:, :int(max_spec_size_entry_Labeling.get())]

                                syllables_spec.append(syllable_spec)

                        LabelingProgress_Text.config(text="Processing...")
                        LabelingProgress_Text.update()
                        LabelingProgress_Bar.config(mode="indeterminate")
                        LabelingProgress_Bar.start()
                        LabelingProgress_Bar.update()
                        gui.update_idletasks()

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
                        mde = pymde.preserve_neighbors(specs_flattened_array, n_neighbors=K,
                                                       embedding_dim=embedding_dim)
                        embedding = mde.embed()

                        # cluster
                        min_cluster_size = math.floor(embedding.shape[0] * float(min_cluster_prop_entry_Labeling.get()))
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1,
                                                    min_samples=int(min_samples_entry_Labeling.get())).fit(embedding)

                        hdbscan_df["labels"] = clusterer.labels_

                        hdbscan_df["X"] = embedding[:, 0]
                        hdbscan_df["Y"] = embedding[:, 1]
                        hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")

                        LabelingProgress_Text.config(text="Saving...")
                        LabelingProgress_Text.update()

                        hdbscan_df.to_csv(output_file)
                        # os.rename(output_file+Bird_ID+"_labels.csv", output_file+Bird_ID+"_labels_"+output_file.split("/")[-1]+".csv")

                        # ------------------------------------

                        LabelingProgress_Text.config(text="Saving...")
                        LabelingProgress_Text.update()

                        # Create UMAP
                        LabelingScatterplot = sns.scatterplot(data=hdbscan_df, x="X", y="Y", hue="labels", alpha=0.25,
                                                              s=5)
                        plt.title("My Bird's Syllables");
                        LabelingFig = LabelingScatterplot.get_figure()

                        # song_folder = song_folder_dir
                        dir_corrected = dir_corrected.replace("_wseg.csv", "_labels.csv")
                        UMAP_Output = output_file.replace("_label_", "_UMAP_")
                        if "_labels.csv" in UMAP_Output:
                            UMAP_Output = UMAP_Output.replace("_labels.csv", "_UMAP-Clusters.png")
                        else:
                            UMAP_Output = UMAP_Output.replace(".csv", "_UMAP-Clusters.png")
                        LabelingFig.savefig(UMAP_Output)

                        UMAP_Directories.append(UMAP_Output)

                        # Generate Metadata File for Advanced Settings #
                        LabelingSettingsMetadata = pd.DataFrame()
                        Labeling_SettingsNames = ["bandpass_lower_cutoff_entry_Labeling",
                                                  "bandpass_upper_cutoff_entry_Labeling", "a_min_entry_Labeling",
                                                  "ref_db_entry_Labeling", "min_level_db_entry_Labeling",
                                                  "n_fft_entry_Labeling", "win_length_entry_Labeling",
                                                  "hop_length_entry_Labeling", "max_spec_size_entry_Labeling"]
                        LabelingSettingsMetadata["Labeling"] = Labeling_SettingsNames

                        Labeling_SettingsValues = pd.DataFrame({'Value': [bandpass_lower_cutoff_entry_Labeling.get(),
                                                                          bandpass_upper_cutoff_entry_Labeling.get(),
                                                                          a_min_entry_Labeling.get(),
                                                                          ref_db_entry_Labeling.get(),
                                                                          min_level_db_entry_Labeling.get(),
                                                                          n_fft_entry_Labeling.get(),
                                                                          win_length_entry_Labeling.get(),
                                                                          hop_length_entry_Labeling.get(),
                                                                          max_spec_size_entry_Labeling.get()]})
                        LabelingUMAP_SettingsNames = pd.DataFrame({'UMAP': ['n_neighbors_entry_Labeling',
                                                                            'n_components_entry_Labeling',
                                                                            'min_dist_entry_Labeling',
                                                                            'spread_entry_Labeling',
                                                                            'metric_entry_Labeling',
                                                                            'random_state_entry_Labeling']})
                        LabelingUMAP_SettingsValues = pd.DataFrame({'UMAP Value': [n_neighbors_entry_Labeling.get(),
                                                                                   n_components_entry_Labeling.get(),
                                                                                   min_dist_entry_Labeling.get(),
                                                                                   spread_entry_Labeling.get(),
                                                                                   metric_variable.get(),
                                                                                   random_state_entry_Labeling.get()]})
                        LabelingClustering_SettingsNames = pd.DataFrame(
                            {'Clustering': ['min_cluster_prop_entry_Labeling', 'spread_entry_Labeling']})
                        LabelingClustering_SettingsValues = pd.DataFrame(
                            {'Clustering Values': [min_cluster_prop_entry_Labeling.get(), spread_entry_Labeling.get()]})

                        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, Labeling_SettingsValues],
                                                             axis=1)
                        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingUMAP_SettingsNames],
                                                             axis=1)
                        LabelingSettingsMetadata = pd.concat([LabelingSettingsMetadata, LabelingUMAP_SettingsValues],
                                                             axis=1)
                        LabelingSettingsMetadata = pd.concat(
                            [LabelingSettingsMetadata, LabelingClustering_SettingsNames],
                            axis=1)
                        LabelingSettingsMetadata = pd.concat(
                            [LabelingSettingsMetadata, LabelingClustering_SettingsValues],
                            axis=1)

                        UMAP_Output = UMAP_Output.replace("_UMAP-Clusters.png", "LabelingSettings_Metadata.csv")
                        LabelingSettingsMetadata.to_csv(UMAP_Output)

                        LabelingProgress_Bar.destroy()
                        LabelingProgress_Text.config(text="Labeling Complete!")
                        LabelingProgress_Text.update()

                        # LabelingDisplay(Direction="Start")

                        # Make folder for storing labeling photos

                        def UMAP_Display(Direction="Start"):
                            global UMAP_Directories
                            global UMAP_CurrentDir
                            if Direction == "Start":
                                UMAP_CurrentDir = 0
                            if Direction == "Next":
                                UMAP_CurrentDir += 1
                            if Direction == "Prev":
                                UMAP_CurrentDir -= 1
                            UMAP_Img = PhotoImage(file=UMAP_Directories[UMAP_CurrentDir])
                            try:
                                UMAP_ImgLabel.destroy()
                            except:
                                pass
                            UMAP_ImgLabel = tk.Label(UMAPWindow, image=UMAP_Img)
                            UMAP_ImgLabel.grid(row=1, columnspan=3)
                            UMAP_ImgLabel.update()
                            UMAPWindow.mainloop()

                        # UMAPWindow = tk.Toplevel(gui)
                        # UMAP_PreviousButton = tk.Button(UMAPWindow, text="Previous",
                        #                                 command=lambda: UMAP_Display("Prev"))
                        # UMAP_PreviousButton.grid(row=0, column=0)
                        # UMAP_NextButton = tk.Button(UMAPWindow, text="Next", command=lambda: UMAP_Display("Next"))
                        # UMAP_NextButton.grid(row=0, column=1)
                        #
                        # EndOfFolder = False
                        # LabelingDisplay("Start")
                        #
                        # UMAP_Display()



        else:
            LabelingErrorMessage.config(text="Invalid segmentation file")
            LabelingErrorMessage.update()

    def LabelingDisplay(Direction):
        global label_FileID
        global OutputFolder
        global hdbscan_df
        global LabelingDisplayFrame
        global LabelingDisplayWindow
        global EndOfFolder
        global label_fig
        global Label_CheckNumber


        if EndOfFolder == True:
            try:
                LabelingDisplayFrame.destroy()
            except: pass

        if Direction == "Start":
            try:
                os.makedirs(OutputFolder + "/LabelingPhotos")
            except: pass
            LabelingDisplayWindow = tk.Toplevel(gui)
            LabelingPreviousButton = tk.Button(LabelingDisplayWindow, text="Previous", command=lambda: LabelingDisplay(Direction = "Left"))
            LabelingPreviousButton.grid(row=0, column=10)
            LabelingNextButton = tk.Button(LabelingDisplayWindow, text="Next", command= lambda: LabelingDisplay(Direction = "Right"))
            LabelingNextButton.grid(row=0, column=11)
            LabelingSaveButton = tk.Button(LabelingDisplayWindow, text="Save", command=lambda: LabelingDisplaySave())
            LabelingSaveButton.grid(row=0, column=12)
            label_FileID = 0

            label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, OutputFolder, "",
                                                                                      song_file_index=label_FileID,
                                                                                      figsize=(12, 4), fontsize=14)

            LabelingDisplayFrame = tk.Frame(LabelingDisplayWindow)
            LabelingDisplayFrame.grid(row=1, column=0, columnspan=21)
            label_canvas = FigureCanvasTkAgg(label_fig, master=LabelingDisplayFrame)  # A tk.DrawingArea.
            label_canvas.draw()
            label_canvas.get_tk_widget().grid()
        if Direction == "Left":
            if label_FileID != 0:
                label_FileID -= 1

                label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, OutputFolder, "",
                                                                                          song_file_index=label_FileID,
                                                                                          figsize=(12, 4), fontsize=14)
                LabelingDisplayFrame = tk.Frame(LabelingDisplayWindow)
                LabelingDisplayFrame.grid(row=1, column=0, columnspan=21)
                label_canvas = FigureCanvasTkAgg(label_fig, master=LabelingDisplayFrame)  # A tk.DrawingArea.
                label_canvas.draw()
                label_canvas.get_tk_widget().grid()
        if Direction == "Right":
            if label_FileID != int(Label_CheckNumber.get()):
                label_FileID += 1

                label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, OutputFolder, "",
                                                                                song_file_index=label_FileID, figsize=(12, 4),fontsize=14)
                LabelingDisplayFrame = tk.Frame(LabelingDisplayWindow)
                LabelingDisplayFrame.grid(row=1, column=0, columnspan=21)
                label_canvas = FigureCanvasTkAgg(label_fig, master=LabelingDisplayFrame)  # A tk.DrawingArea.
                label_canvas.draw()
                label_canvas.get_tk_widget().grid()
        global FileName
        FileName = hdbscan_df["files"].unique()[label_FileID][:-4]
        def LabelingDisplayClose():
            try:
                shutil.rmtree(OutputFolder + "/LabelingPhotos")
            except:
                pass
            try:
                shutil.rmtree(OutputFolder + "/Labeling_temp")
            except:
                pass

        LabelingDisplayWindow.protocol("WM_DELETE_WINDOW", LabelingDisplayClose())

        LabelingDisplayWindow.mainloop()

    def LabelingDisplaySave():
        global label_fig
        global LabelingBirdIDText
        global FileName

        f = asksaveasfile(initialfile=str(FileName)+'_LabeledSpectrogram.png',defaultextension=".png", filetypes=[("PNG Image", "*.png"),("All Files", "*.*")])

    def LabelingSavePhotos():
        global LabelingSaveAllCheck
        global label_FileID
        global labeling_path
        global BulkLabelFiles_to_save
        song_folder = ""

        pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')

        if "label" not in labeling_path:
            if LabelingBirdIDText.get() != "Bird ID" and pattern.search(LabelingBirdIDText.get()) == None:
                LabelingBulkSaveError.config(text="Invalid Labeling File and Bird ID")
            else:
                LabelingBulkSaveError.config(text="Invalid Labeling File")
        elif LabelingBirdIDText.get() != "Bird ID" and pattern.search(LabelingBirdIDText.get()) == None:
            LabelingBulkSaveError.config(text="Invalid Bird ID")
        else:
            for i in labeling_path.split("/")[:-1]:
                song_folder = song_folder+i+"/"
            hdbscan_df = pd.read_csv(labeling_path)
            LabelingSaveProgress = tk.Label(LabelingBulkSave, text="Saving...")
            LabelingSaveProgress.grid(row=6, column=0, columnspan=2)
            time.sleep(0.1)
            try:
                os.makedirs(song_folder+"/LabelingPhotos")
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
            time.sleep(3)
            LabelingSaveProgress.destroy()

    def LabelingCheckSpectrograms():
        global segmentations_path
        global Bird_ID
        global song_folder
        global LabelingFileDisplay
        global LabelingErrorMessage
        global LabelingBirdIDText
        global UMAP_Directories
        global OutputFolder
        global Label_CheckNumber

        FilesToCheck = int(Label_CheckNumber.get())
        OutputFolder = LabelingOutputFile_Text.cget("text")
        for directory in segmentations_path:
            directory = directory.replace("\\", "/")

            dir_corrected = directory.split("/")[-1].replace("_seg_", "_label_")
            dir_corrected = dir_corrected.replace("_wseg.csv", str(directory.split("/")[-2]) + "_labels.csv")
            try:
                os.makedirs(OutputFolder + "/Labeling_temp")
            except:
                pass
            output_file = OutputFolder + "/Labeling_temp/" + dir_corrected  # e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"

            def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
                spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length,
                                           n_fft=n_fft)
                spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

                # normalize
                S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

                return S_norm

            amin = 1e-5
            ref_db = 20
            min_level_db = -28  # ERROR! If a .wav file contains a sound that's too quiet, the audio.values variable for that row will be NULL, causing an error
            win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
            hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
            n_fft = 512
            K = 10
            min_cluster_prop = 0.04
            embedding_dim = 2

            # load segmentations
            global LabelingDirectoryText
            segmentations_temp = pd.read_csv(LabelingDirectoryText.cget("text"))
            segmentations_temp = segmentations_temp.rename(
                columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
            UniqueFiles_temp = pd.unique(segmentations_temp["files"])
            UniqueFiles = UniqueFiles_temp[:FilesToCheck]
            segmentations = segmentations_temp[segmentations_temp["files"].isin(UniqueFiles)]

            # sometimes this padding can create negative onset times, which will cause errors.
            # correct this by setting any onsets <0 to 0.
            segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

            # Add syllable audio to dataframe
            syllable_dfs = pd.DataFrame()
            # for directory in segmentations.directory.unique():

            for song_file in segmentations.files.unique():
                SongIndex = segmentations.index[segmentations['files'] == song_file].tolist()
                file_path = segmentations.directory[SongIndex[0]].replace("\\", "/") + song_file

                song = dataloading.SongFile(file_path)
                song.bandpass_filter(int(bandpass_lower_cutoff_entry_Labeling.get()),
                                     int(bandpass_upper_cutoff_entry_Labeling.get()))

                syllable_df = segmentations[segmentations['files'] == song_file]
                # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
                syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                        for st, et in
                                        zip(syllable_df.onsets.values, syllable_df.offsets.values)]
                # print(syllable_df)
                syllable_dfs = pd.concat([syllable_dfs, syllable_df])

            syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
            global hdbscan_df
            hdbscan_df = syllable_dfs

            # compute spectrogram for each syllable
            syllables_spec = []

            for syllable in syllable_dfs.audio.values:
                if len(syllable) > 0: \
                        syllable_spec = make_spec(syllable,
                                                  hop_length=int(hop_length_entry_Labeling.get()),
                                                  win_length=int(win_length_entry_Labeling.get()),
                                                  n_fft=int(n_fft_entry_Labeling.get()),
                                                  ref_db=int(ref_db_entry_Labeling.get()),
                                                  amin=float(a_min_entry_Labeling.get()),
                                                  min_level_db=int(min_level_db_entry_Labeling.get()))
                if syllable_spec.shape[1] > int(max_spec_size_entry_Labeling.get()):
                    print(
                        "Long Syllable Corrections! Spectrogram Duration = " + str(
                            syllable_spec.shape[1]))
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
            mde = pymde.preserve_neighbors(specs_flattened_array, n_neighbors=K,
                                           embedding_dim=embedding_dim)
            embedding = mde.embed()

            # cluster
            min_cluster_size = math.floor(embedding.shape[0] * float(min_cluster_prop_entry_Labeling.get()))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1,
                                        min_samples=int(min_samples_entry_Labeling.get())).fit(embedding)

            hdbscan_df["labels"] = clusterer.labels_

            hdbscan_df["X"] = embedding[:, 0]
            hdbscan_df["Y"] = embedding[:, 1]
            hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")

            output_file = OutputFolder + "/Labeling_temp/" + dir_corrected
            hdbscan_df.to_csv(output_file)
        global EndOfFolder
        EndOfFolder = False
        LabelingDisplay("Start")

    def LabelingSpectrograms():
        global LabelingSpectrogramFiles
        global LabelingFileDisplay
        # LabelingFile = LabelingFileDisplay.cget("text")
        LabelingFile = LabelingSpectrogramFiles

        for file in LabelingFile:
            LabelingData = pd.read_csv(file)
            Split_LabelingFile = file.split("/")[:-1]
            Merged_LabelingFile = ""
            for a in Split_LabelingFile:
                Merged_LabelingFile = Merged_LabelingFile + a + "/"
            LabelingOutputDir = os.path.join(Merged_LabelingFile,
                                             "Labeled_Spectrograms_" + str(file.split("/")[-1].split(".")[0]))
            try:
                os.makedirs(LabelingOutputDir)
            except:
                pass
            global MaxFiles_Labeling
            MaxFiles = int(MaxFiles_Labeling.get())
            if len(LabelingData["files"]) < MaxFiles:
                i_end = len(LabelingData["files"])
            else:
                i_end = MaxFiles
            for i in range(i_end):
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(LabelingData,
                                                                                    LabelingData["directory"][
                                                                                        i].replace("\\", ""), "",
                                                                                    song_file_index=i,
                                                                                    figsize=(15, 4),
                                                                                    fontsize=14)
                # print(LabelingOutputDir)
                fig.savefig(LabelingOutputDir + "/" + str(song_file_name) + ".png")
                plt.close(fig)
            print("Done!")

    def Acoustics_Interval():
        global ContCalc
        AcousticsProgress = tk.Label(AcousticsMainFrameSingle, text="Calculating Acoustics...")
        AcousticsProgress.grid(row=12, column=1)
        global AcousticsDirectory

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


        global AcousticsOutput_Text
        if ContCalc.get() == 0:
            song = dataloading.SongFile(AcousticsDirectory)
            if AcousticsOffset.get() == "End of File":
                song_interval = acoustics.SongInterval(song, onset=float(AcousticsOnset.get()), offset=None)
            else:
                song_interval = acoustics.SongInterval(song, onset=float(AcousticsOnset.get()),
                                                       offset=float(AcousticsOffset.get()))
            #song_interval.calc_all_feature_stats(features=FeatureList)
            #song_interval.save_features(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/",file_name=str(AcousticsDirectory.split("/")[-1][:-4]))
            #feature_stats = song_interval.calc_feature_stats(features=FeatureList)
            #print(feature_stats)
            song_interval.save_feature_stats(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/", file_name="Stats_" + str(AcousticsDirectory.split("/")[-1][:-4]), features=FeatureList)
            AcousticsProgress.config(text="Acoustics Calculations Complete!")
            print(str(AcousticsOutput_Text.cget("text"))+"/")
        if ContCalc.get() == 1:
            AllSongFiles = glob.glob(str(AcousticsOutput_Text.cget("text")) + "/**/*.wav", recursive=True)
            #song_features = song_interval.calc_all_features(features=FeatureList)
            try:
                os.makedirs(AcousticsOutput_Text.cget("text")+"/AcousticsOutput")
            except:
                pass
            for file in AllSongFiles:
                song = dataloading.SongFile(file)
                audioread.audio_open(file).duration
                song_interval = acoustics.SongInterval(song, onset=0, offset=None)
                song_interval.save_features(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/AcousticsOutput/",file_name=str((file.split("/")[-1][:-4]).split("\\")[-1]), features=FeatureList)

    def Acoustics_Syllables():
        global MultiAcousticsDirectory
        global Bird_ID

        if "seg" not in MultiAcousticsDirectory:
            MultiAcousticsError.config(text="Invalid Segmentation File")
            MultiAcousticsError.update()
        else:
            # Check that at least one feature is selected
            if (MultiRunGoodness.get()+MultiRunMean_frequency.get()+MultiRunEntropy.get()+MultiRunAmplitude.get()+
                MultiRunAmplitude_modulation.get()+MultiRunFrequency_modulation.get()+MultiRunPitch.get()) == 0:
                    MultiAcousticsError.config(text="No Features Selected")
                    MultiAcousticsError.update()
            else:
                MultiAcousticsProgress = tk.Label(AcousticsMainFrameMulti, text="Calculating Acoustics...")
                MultiAcousticsProgress.grid(row=25, column=1)
                MultiAcousticsProgress.update()
                time.sleep(1)

                syll_df = pd.read_csv(str(MultiAcousticsDirectory))
                syll_df=syll_df.rename(columns={"onset":"onsets","offset":"offsets", "file":"files"})
                global MultiAcousticsOutputDisplay
                MultiAcousticsOutputDirectory_temp = MultiAcousticsOutputDisplay.cget("text")
                try:
                    os.makedirs(MultiAcousticsOutputDirectory_temp+"/Acoustics_WholeFolder/")
                    MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory_temp+"/Acoustics_WholeFolder/"
                except:
                    MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory_temp+"/Acoustics_WholeFolder/"


                # MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory + "/"

                MultiAcousticsDirectory2 = ""
                for a in MultiAcousticsDirectory.split("/")[:-1]:
                    MultiAcousticsDirectory2 = MultiAcousticsDirectory2+a+"/"
                MultiAcousticsDirectory = MultiAcousticsDirectory2

                # Try below function, assuming BirdID is chosen, otherwise grab bird ID from directory and proceed
                try:
                    acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=MultiAcousticsDirectory,
                                                           win_length=int(win_length_entry_Acoustics.get()),hop_length= int(hop_length_entry_Acoustics.get()),
                                                           n_fft=int(n_fft_entry_Acoustics.get()), max_F0=int(max_F0_entry_Acoustics.get()),min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                           freq_range=float(freq_range_entry_Acoustics.get()),baseline_amp=int(baseline_amp_entry_Acoustics.get()),fmax_yin=int(fmax_yin_entry_Acoustics.get()))
                except:
                    pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
                    Bird_ID = str(pattern.search(MultiAcousticsDirectory).group())
                    acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=MultiAcousticsDirectory,win_length=int(win_length_entry_Acoustics.get()),hop_length= int(hop_length_entry_Acoustics.get()),
                                                           n_fft=int(n_fft_entry_Acoustics.get()), max_F0=int(max_F0_entry_Acoustics.get()),min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                           freq_range=float(freq_range_entry_Acoustics.get()),baseline_amp=int(baseline_amp_entry_Acoustics.get()),fmax_yin=int(fmax_yin_entry_Acoustics.get()))

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
                AcousticSettingsMetadata.to_csv(MultiAcousticsOutputDirectory + "/"+str(Bird_ID)+"_AcousticSettings_Metadata.csv")

                MultiAcousticsProgress.config(text="Acoustics Calculations Complete!")

    def Syntax():
        import avn.syntax as syntax
        import avn.plotting as plotting
        global SyntaxDirectory
        global DropCalls
        global SyntaxFileDisplay
        SyntaxProgress = tk.Label(SyntaxFrame, text="Running...")
        SyntaxProgress.grid(row=5, column=0)
        Bird_ID = SyntaxBirdID.get()
        syll_df = pd.read_csv(SyntaxFileDisplay.cget("text"))
        global syntax_data
        Bird_ID = ""

        merged_syntax_data = pd.DataFrame()
        Syntax_DirIndex = 0
        for directory in syll_df["directory"].unique():
            Syntax_DirIndex+=1
            global SyntaxOutputDisplay
            try:
                SyntaxOutputFolder = SyntaxOutputDisplay.cget("text") + "/Syntax_" + SyntaxBirdID.get() + "_Day"+str(Syntax_DirIndex)+"/"
                os.makedirs(SyntaxOutputFolder)
            except:
                pass
            # except:
            #     dir_list_temp = glob.glob(SyntaxOutputDisplay.cget("text") + "/Syntax_" + SyntaxBirdID.get() + "*")
            #     dir_list = []
            #     for dir in dir_list:
            #         if ".csv" not in dir:
            #             dir_list.append(dir)
            #     os.makedirs(SyntaxOutputDisplay.cget("text") + "/Syntax_" + SyntaxBirdID.get() + "_" + str(
            #         int(dir_list.replace("\\", "/").split("_")[-1]) + 1))
            #     SyntaxOutputFolder = SyntaxOutputDisplay.cget("text") + "/Syntax_" + SyntaxBirdID.get() + "_Day" + str(
            #         int(dir_list[-1][-1]) + 1)
            # Prev_Iters = glob.glob(SyntaxOutputFolder+ "/*.csv")
            # if Prev_Iters != []:
            #     for file in Prev_Iters:
            #         if "temp" not in file:
            #             os.rename(file, file[:-4]+"_temp.csv")

            # Make dataframe for current directory
            syll_df_temp = syll_df[syll_df["directory"] == directory]
            syntax_data = syntax.SyntaxData(Bird_ID, syll_df_temp)
            dir_corrected = directory.replace("\\", "/")[:-1]
            syntax_data.add_file_bounds(dir_corrected)
            syntax_data.add_gaps(min_gap=float(min_gap_entry_Syntax.get()))
            gaps_df = syntax_data.get_gaps_df()
            if DropCalls.get() == 1:
                syntax_data.drop_calls()
            syntax_data.make_transition_matrix()
            entropy_rate = syntax_data.get_entropy_rate()
            entropy_rate_norm = entropy_rate / np.log2(len(syntax_data.unique_labels) + 1)
            prob_repetitions = syntax_data.get_prob_repetitions()
            single_rep_counts, single_rep_stats = syntax_data.get_single_repetition_stats()
            intro_notes_df = syntax_data.get_intro_notes_df()
            prop_sylls_in_short_bouts = syntax_data.get_prop_sylls_in_short_bouts(max_short_bout_len=2)
            per_syll_stats = syntax.Utils.merge_per_syll_stats(single_rep_stats, prop_sylls_in_short_bouts, intro_notes_df)
            pair_rep_counts, pair_rep_stats = syntax_data.get_pair_repetition_stats()
            tempSyntaxDirectory2 = ""

            # syntax_analysis_metadata = syntax_data.save_syntax_data(tempSyntaxDirectory2)
            #syntax_data["directory"] = np.array(len(syntax_data["directory"]*directory))

            # Prev_df = pd.read_csv(SyntaxOutputFolder+"/temp.csv")
            # merged_syntax_data.save_syntax_data(SyntaxOutputFolder)
            # Curr_df = pd.read_csv(SyntaxOutputFolder+"")
            syntax_data.save_syntax_data(SyntaxOutputFolder)
            temp_file_list = []
            all_files_list = glob.glob(SyntaxOutputFolder+ "/*.csv")

            # if len(all_files_list) > 0:
            #     for file in all_files_list:
            #         if "temp" in file:
            #             temp_file_list.append(file)
            #         if "syll_df_temp.csv" in file:
            #             old_syll_df = file
            #         if "syll_df.csv" in file:
            #             new_syll_df = file
            #         if "syntax_analysis_metadata_temp.csv" in file:
            #             old_syntax_analysis_metadata = file
            #         if "syntax_analysis_metadata.csv" in file:
            #             new_syntax_analysis_metadata = file
            #         if "trans_mat_temp.csv" in file:
            #             old_trans_mat = file
            #         if "trans_mat.csv" in file:
            #             new_trans_mat = file
            #         if "trans_mat_prob_temp.csv" in file:
            #             old_trans_mat_prob = file
            #         if "trans_mat_prob.csv" in file:
            #             new_trans_mat_prob = file
            #
            #     merged_syll_df = pd.concat([old_syll_df, new_syll_df])
            #     merged_syntax_analysis_metadata = pd.concat([old_syntax_analysis_metadata, new_syntax_analysis_metadata])
            #     merged_trans_mat = pd.concat([old_trans_mat, new_trans_mat])
            #     merged_trans_mat_prob = pd.concat([old_trans_mat_prob, new_trans_mat_prob])
            #
            #     for file in all_files_list:
            #         try:
            #             os.remove(file)
            #         except: pass
            try:
                merged_syll_df.to_csv(SyntaxOutputFolder+"syll_df.csv")
                merged_syntax_analysis_metadata.to_csv(SyntaxOutputFolder+"syntax_analysis_metadata.csv")
                merged_trans_mat.to_csv(SyntaxOutputFolder+"trans_mat.csv")
                merged_trans_mat_prob.to_csv(SyntaxOutputFolder+"trans_mat_prob.csv")
            except:
                syntax_data.save_syntax_data(SyntaxOutputFolder)
        SyntaxProgress.config(text="Complete!")

    def PrintPlainSpectrograms():
        global PlainDirectory
        try:
            os.makedirs(PlainDirectory + "/Unlabeled Spectrograms")
        except: pass
        FileList = glob.glob(str(PlainDirectory) + "/*.wav")
        PlainProgressLabel = tk.Label(PlainSpectrograms, text="Generating Spectrograms...")
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

    def Timing():
        syll_df = pd.read_csv(TimingInput_Text.cget("text"))
        Bird_ID = ""
        syll_df = syll_df.rename(
            columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
        try:
            os.makedirs(TimingOutput_Text.cget("text")+"/Timing/")
        except:
            pass
        Temp_TimingOutputFolder = TimingOutput_Text.cget("text")+"/Timing/"
        DirectoryCount = 0

        Timing_df = pd.DataFrame({"directory":[],"syll_duration_entropy":[],"gap_duration_entropy":[]})

        for directory in syll_df["directory"].unique():
            DirectoryCount +=1
            Temp_Array = []
            Temp_Array.append(directory)
            TimingOutputFolder = Temp_TimingOutputFolder+"Day"+str(DirectoryCount)+"/"
            try:
                os.makedirs(TimingOutputFolder)
            except:
                pass
            syll_df_temp = syll_df[syll_df["directory"]==directory]
            segment_timing = avn.timing.SegmentTiming(Bird_ID, syll_df_temp,
                                                      song_folder_path=directory)
            syll_durations = segment_timing.get_syll_durations()
            syll_durations.to_csv(TimingOutputFolder+"syll_durations.csv")
            print("a")

            # There seems to be a conflict b/w seaborn (sns) and pandas, so the sns plotting functions raise an error... Everything else works fine
            sns.kdeplot(data=syll_durations, x='durations', bw_adjust=0.1)
            plt.title('Syllable Durations')
            plt.xlabel('Syllable duration (s)');
            plt.savefig(TimingOutputFolder + "fig1.png")
            syll_duration_entropy = segment_timing.calc_syll_duration_entropy()
            Temp_Array.append(syll_duration_entropy)
            gap_durations = segment_timing.get_gap_durations(max_gap=0.2)
            sns.kdeplot(data=gap_durations, x='durations', bw_adjust=0.1)

            print("b")

            plt.title('Gap Durations')
            plt.xlabel('Gap duration (s)');
            plt.savefig(TimingOutputFolder + "fig2.png")

            gap_duration_entropy = segment_timing.calc_gap_duration_entropy()
            Temp_Array.append(gap_duration_entropy)

            rhythm_analysis = avn.timing.RhythmAnalysis("")
            song_folder_path = ""
            for i in TimingInput_Text.cget("text").split("/")[:-1]:
                song_folder_path = song_folder_path+i+"/"
            rhythm_spectrogram = rhythm_analysis.make_rhythm_spectrogram(
                song_folder_path=song_folder_path)
            print(TimingInput_Text.cget("text"))
            print("c")

            # For some reason, the rhythm spectrogram is empty...
            print(rhythm_analysis.rhythm_spectrogram)

            fig_rhythm_spectrogram = rhythm_analysis.plot_rhythm_spectrogram()
            fig_rhythm_spectrogram.savefig(TimingOutput_Text.cget("text") + "fig3.png")
            rhythm_spectrogram_entropy = rhythm_analysis.calc_rhythm_spectrogram_entropy()
            peak_frequencies = rhythm_analysis.get_refined_peak_frequencies(freq_range=3)
            fig_peak_frequencies = rhythm_analysis.plot_peak_frequencies()
            fig_peak_frequencies.savefig(TimingOutputFolder + "fig4.png")
            peak_frequency_cv = rhythm_analysis.calc_peak_frequency_cv()

            Timing_df.loc[len(Timing_df.index)] = Temp_Array

        Timing_df.to_csv(TimingOutputFolder+"Timing_Stats.csv")
        print("done")

    def MoreInfo(Event):
        print(Event.widget)
        Text_wraplength = 300

        SettingsDict = {
                        "10": {"2":"Lower cutoff frequency in Hz for a hamming window " \
                                   "bandpass filter applied to the audio data before generating " \
                                   "spectrograms. Frequencies below this value will be filtered out"+'!bandpass_lower_cutoff',
                            "4":"Upper cutoff frequency in Hz for a hamming window bandpass" \
                                   " filter applied to the audio data before generating spectrograms. " \
                                   "Frequencies above this value will be filtered out"+"!bandpass_upper_cutoff",
                            "6":"Minimum amplitude threshold in the spectrogram. Values " \
                                   "lower than a_min will be set to a_min before conversion to decibels"+"!a_min",
                            "8":"When making the spectrogram and converting it from amplitude " \
                                   "to db, the amplitude is scaled relative to this reference: " \
                                   "20 * log10(S/ref_db) where S represents the spectrogram with amplitude values"+"!ref_db",
                            "10":"When making the spectrogram, once the amplitude has been converted " \
                                   "to decibels, the spectrogram is normalized according to this value: " \
                                   "(S - min_level_db)/-min_level_db where S represents the spectrogram " \
                                   "in db. Any values of the resulting operation which are <0 are set to " \
                                   "0 and any values that are >1 are set to 1"+"!min_level_db",
                            "12":"When making the spectrogram, this is the length of the windowed " \
                                   "signal after padding with zeros. The number of rows spectrogram is" \
                                   " \"(1+n_fft/2)\". The default value,\"n_fft=512\" samples, " \
                                   "corresponds to a physical duration of 93 milliseconds at a sample " \
                                   "rate of 22050 Hz, i.e. the default sample rate in librosa. This value " \
                                   "is well adapted for music signals. However, in speech processing, the " \
                                   "recommended value is 512, corresponding to 23 milliseconds at a sample" \
                                   " rate of 22050 Hz. In any case, we recommend setting \"n_fft\" to a " \
                                   "power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm"+"!n_fft",
                            "14":"When making the spectrogram, each frame of audio is windowed by a window " \
                                   "of length \"win_length\" and then padded with zeros to match \"n_fft\"." \
                                   " Padding is added on both the left- and the right-side of the window so" \
                                   " that the window is centered within the frame. Smaller values improve " \
                                   "the temporal resolution of the STFT (i.e. the ability to discriminate " \
                                   "impulses that are closely spaced in time) at the expense of frequency " \
                                   "resolution (i.e. the ability to discriminate pure tones that are closely" \
                                   " spaced in frequency). This effect is known as the time-frequency " \
                                   "localization trade-off and needs to be adjusted according to the " \
                                   "properties of the input signal"+"!win_length",
                            "16":"The number of audio samples between adjacent windows when creating " \
                                   "the spectrogram. Smaller values increase the number of columns in " \
                                   "the spectrogram without affecting the frequency resolution"+"!hop_length",
                            "18":"Maximum frequency in Hz used to estimate fundamental frequency " \
                                   "with the YIN algorithm"+"!max_spec_size"},
                        "11": {"2":"The size of local neighborhood (in terms of number of neighboring sample points)" \
                                   " used for manifold approximation. Larger values result in more global views " \
                                   "of the manifold, while smaller values result in more local data being " \
                                   "preserved. In general values should be in the range 2 to 100"+'!n_neighbors',
                              "4":"The dimension of the space to embed into. This defaults to 2 to provide " \
                                   "easy visualization, but can reasonably be set to any integer value " \
                                   "in the range 2 to 100"+"!n_components",
                              "6":"The effective minimum distance between embedded points. " \
                                   "Smaller values will result in a more clustered/clumped " \
                                   "embedding where nearby points on the manifold are drawn " \
                                   "closer together, while larger values will result on a more " \
                                   "even dispersal of points. The value should be set relative to " \
                                   "the \"spread\" value, which determines the scale at which " \
                                   "embedded points will be spread out"+'!min_dist',
                              "8":"The effective scale of embedded points. In combination with " \
                                   "\"min_dist\" this determines how clustered/clumped the embedded points are"+'!spread',
                              "10":"The metric to use to compute distances in high dimensional space"+'!metric',
                              "12":"If specified, random_state is the seed used by the random " \
                                   "number generator. Specifying a random state is the only way " \
                                   "to ensure that you can reproduce an identical UMAP with the " \
                                   "same data set multiple times"+'!random_state'},
                        "12": {"2":'Minimum fraction of syllables that can constitute a cluster. '
                                              'For example, in a dataset of 1000 syllables, there need to be '
                                              'at least 40 instances of a particular syllable for that to be '
                                              'considered a cluster. Single linkage splits that contain fewer '
                                              'points than this will be considered points “falling out” of a '
                                              'cluster rather than a cluster splitting into two new clusters'
                                              'when performing HDBSCAN clustering'+"!min_cluster_prop",
                              "4":'The number of samples in a neighbourhood for a point to be '
                                              'considered a core point in HDBSCAN clustering. The larger the '
                                              'value of \"min_samples\" you provide, the more conservative '
                                              'the clustering – more points will be declared as noise, and '
                                              'clusters will be restricted to progressively more dense areas'+"!min_samples"},
                        "16": {"2":"Length of window over which to calculate each feature in samples"+'!win_length',
                            "4":"Number of samples to advance between windows"+'!hop_length',
                            "6":"Length of the transformed axis of the output. If n is smaller than " \
                                               "the length of the win_length, the input is cropped"+'!n_fft',
                            "8":"Maximum allowable fundamental frequency of signal in Hz"+'!max_F0',
                            "10":"Lower frequency cutoff in Hz. Only power at frequencies above " \
                                               "this will contribute to feature calculation"+'!min_frequency',
                            "12":"Proportion of power spectrum frequency bins to consider"+'!freq_range',
                            "14":"Baseline amplitude used to calculate amplitude in dB"+'!baseline_amp',
                            "16":"Maximum frequency in Hz used to estimate fundamental frequency " \
                                               "with the YIN algorithm"+'!fmax_yin'},
                        "19": {"2":"Minimum duration in seconds for a gap between syllables "
                                     "to be considered syntactically relevant. This value should "
                                     "be selected such that gaps between syllables in a bout are "
                                     "shorter than min_gap, but gaps between bouts are longer than min_gap"+"!min_gap"}
        }

        Module = str(Event.widget).split(".")[1].split("e")[-1]
        Setting = str(Event.widget).split("n")[-1]

        ### Labeling Spectrogram Generation Parameters ###
        if Module == "10":
            LabelingSpectrogramDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
            LabelingSpectrogramDialog.update()
            LabelingSpectrogramTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
            LabelingSpectrogramTitle.update()

        ### UMAP Parameters ###
        if Module == "11":
            LabelingUMAPDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
            LabelingUMAPDialog.update()
            LabelingUMAPTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
            LabelingUMAPTitle.update()

        ### Clustering Parameters ###
        if Module == "12":
            LabelingClusterDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
            LabelingClusterDialog.update()
            LabelingClusterTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
            LabelingClusterTitle.update()

        ### Acoustic Features ###
        if Module == "16":
            AcousticSettingsDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
            AcousticSettingsDialog.update()
            AcousticsSettingsDialogTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
            AcousticsSettingsDialogTitle.update()

        ### Syntax ###
        if Module == "19":
            SyntaxDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
            SyntaxDialog.update()
            SyntaxDialogTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
            SyntaxDialogTitle.update()

    def LessInfo(Event):
        pass
        # if "frame11" in str(Event.widget):
        #     AcousticSettingsDialog.config(text="")
        #     AcousticsSettingsDialogTitle.config(text="")
        # if "frame4" in str(Event.widget):
        #     LabelingSpectrogramDialog.config(text="")
        #     LabelingSpectrogramTitle.config(text="")
        # if "frame5" in str(Event.widget):
        #     LabelingUMAPDialog.config(text="")
        #     LabelingUMAPTitle.config(text="")
        # if "frame6" in str(Event.widget):
        #     LabelingClusterDialog.config(text="")
        #     LabelingClusterTitle.config(text="")

    def Validate_Settings(Widget, WidgetName, ErrorLabel):
        IntVarList = ['n_components_entry_Labeling', 'random_state_entry_Labeling', 'min_samples_entry_Labeling']
        FloatVarList = ['a_min_entry_Labeling', 'min_dist_entry_Labeling', 'min_gap_entry_Syntax', 'spread_entry_Labeling']
        NormalList = ['bandpass_lower_cutoff_entry_Labeling', 'bandpass_upper_cutoff_entry_Labeling', 'ref_db_entry_Labeling', 'min_level_db_entry_Labeling','n_fft_entry_Labeling', 'win_length_entry_Labeling', 'hop_length_entry_Labeling', 'max_spec_size_entry_Labeling']
        # min_cluster_prop must be between 0 and 1
        # n_neighbors should be b/w 2 and 100

        MergedIntList = IntVarList+NormalList
        if WidgetName in MergedIntList:
            try:
                int(Widget.get())
                ErrorLabel.config(text="")
                Widget.config(bg="white")
            except:
                ErrorLabel.config(text="Invalid Input: Please Enter An Integer")
                Widget.config(bg="red")
        elif WidgetName in FloatVarList:
            try:
                float(Widget.get())
                ErrorLabel.config(text="")
                Widget.config(bg="white")
            except:
                ErrorLabel.config(text="Invalid Input: Please Enter A Number")
                Widget.config(bg="red")
        elif WidgetName == "min_cluster_prop_entry_Labeling":
            try:
                temp_float = float(Widget.get())
                if temp_float > 1 or temp_float < 0:
                    ErrorLabel.config(text="Outside of Valid Range (0 - 1)")
                    Widget.config(bg="red")
                else:
                    ErrorLabel.config(text="")
                    Widget.config(bg="white")
            except:
                ErrorLabel.config(text="Invalid Input: Please Enter A Number")
                Widget.config(bg="red")
        elif WidgetName == "n_neighbors_entry_Labeling":
            try:
                temp_int = int(Widget.get())
                if temp_int > 100 or temp_int < 2:
                    ErrorLabel.config(text="Outside of Valid Range (2 - 100)")
                    Widget.config(bg="red")
                else:
                    ErrorLabel.config(text="")
                    Widget.config(bg="white")
            except:
                ErrorLabel.config(text="Invalid Input: Please Enter An Integer")
                Widget.config(bg="red")
        else:
            try:
                int(Widget.get())
                ErrorLabel.config(text="")
                Widget.config(bg="white")
            except:
                ErrorLabel.config(text="Invalid Input: Please Enter An Integer")
                Widget.config(bg="red")

    # def Extra_Labeling():
    #     global LabelingExtra_Input_Text
    #     # global LabelingExtra_Output_Text
    #     global LabelingBirdIDText
    #     global segmentations_path
    #     global LabelingOutputFile_Text
    #     global LoadingBar_Massimo
    #
    #     # LabelingBirdIDText.config(text=LabelingExtra_BirdID)
    #     # LabelingBirdIDText.update()
    #     Seg_File_List = glob.glob(LabelingExtra_Input_Text.cget("text")+"/**/*.csv", recursive=True)
    #     print(Seg_File_List)
    #     Master_Output = pd.DataFrame()
    #     LoadingBar_Massimo = ttk.Progressbar(LabelingMainFrame, mode="determinate", maximum=len(Seg_File_List), orient="horizontal")
    #     LoadingBar_Massimo.grid(row=11)
    #     gui.update_idletasks()
    #     segmentations_path = tuple(Seg_File_List)
    #     Labeling()
    #
    #     for file in Seg_File_List:
    #         # temp_dir = LabelingOutputFile_Text.cget("text").replace("\\","/").split("/")[:-1]
    #         # Out_dir = ""
    #         # for a in temp_dir:
    #         #     Out_dir = Out_dir+a+"/"
    #         # LabelingOutputFile_Text.config(text=Out_dir)
    #         # LabelingOutputFile_Text.update()
    #         Out_dir = LabelingOutputFile_Text.cget("text")
    #
    #         LoadingBar.step(1)
    #         LoadingBar.update()
    #         dir_files = glob.glob(Out_dir+"/Labeling/")
    #         print("-----")
    #         for csv in dir_files:
    #             print(csv)
    #             Label_Data = pd.read_csv(csv)
    #             counts = Label_Data.value_counts("labels")
    #             print(counts)
    #             Output_df = pd.DataFrame(counts)
    #             Master_Output.join([Master_Output, Output_df])
    #     # print(LabelingExtra_Output_Text.cget("text"))
    #     Master_Output.to_csv(LabelingOutputFile_Text.cget("text")+"Master_Output.csv")
    #     print("done!")

    ### Initialize gui ###
    gui = tk.Tk()
    gui.title("AVN GUI")
    gui.resizable(False, False)

    ParentStyle = ttk.Style()
    ParentStyle.configure('Parent.TNotebook.Tab', font=('Arial', '10', 'bold'))

    notebook = ttk.Notebook(gui, style='Parent.TNotebook.Tab')
    notebook.grid()

    MasterFrameWidth = 1000
    MasterFrameHeight = 500

    Dir_Width = 35
    BirdID_Width = 10
    Padding_Width = 10

    ### Home Tab ###
    HomeTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(HomeTab, text="Home")
    HomeNotebook = ttk.Notebook(HomeTab)
    HomeNotebook.grid()
    WelcomeTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    HomeNotebook.add(WelcomeTab, text="Welcome!")
    MoreInfoTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    HomeNotebook.add(MoreInfoTab, text="More Info")
    LinksTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    HomeNotebook.add(LinksTab, text="Helpful Links")

    # Welcome Tab #
    WelcomeMessage = tk.Label(WelcomeTab, text="Welcome to the AVN GUI! \n \n If this is your first time using the GUI,\n please "
                                               "refer to the \"More Info\" tab \n to get started.", font=("Arial", 20)).grid()

    # More Info Tab #
    More_Info_Title = tk.Label(MoreInfoTab, text="What does this GUI do?", font=("Arial", 20,"bold")).grid(row=0, column=0, columnspan=3)
    More_Info_Body = tk.Label(MoreInfoTab, text="Labeling:"+
                                                "\nAcoustic Feature Calculations:"+
                                                "\nSyntax Analysis:"+
                                                "\nTiming Analysis:", justify="left").grid(row=1, column=0)

    # Helpful Links Tab #
    HelpfulLinks = tk.Label(LinksTab, text="Example hyperlink", fg="blue", cursor="hand2")
    HelpfulLinks.grid()
    HelpfulLinks.bind("<Button-1>", lambda e: wb.open_new_tab("http://www.google.com"))


    ### Labeling Window Initialization ###
    # Labeling Features
    LabelingFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(LabelingFrame, text="Labeling")
    LabelingMainFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook = ttk.Notebook(LabelingFrame)
    LabelingNotebook.grid(row=1)
    LabelingNotebook.add(LabelingMainFrame, text="Home")
    LabelingSpectrogram_UMAP = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingSpectrogram_UMAP, text="Generate Spectrograms")

    LabelingBulkSave = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingBulkSave, text="Save Many Spectrograms")

    LabelingSettingsMaster = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingNotebook.add(LabelingSettingsMaster, text="Settings")

    # Labeling Settings
    LabelingSettingsNotebook = ttk.Notebook(LabelingSettingsMaster, style='Parent.TNotebook.Tab')
    LabelingSettingsNotebook.grid()
    LabelingSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingSettingsNotebook.add(LabelingSettingsFrame, text="Spectrogram Settings")
    LabelingUMAPSettings = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingSettingsNotebook.add(LabelingUMAPSettings, text="UMAP Settings")
    LabelingClusterSettings = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    LabelingSettingsNotebook.add(LabelingClusterSettings, text="Clustering Settings")

###############################################
    # Main Labeling Window
    Labeling_SpectrogramCheck = tk.Button(LabelingMainFrame, text="Check Spectrograms", command=lambda:LabelingCheckSpectrograms())
    Labeling_SpectrogramCheck.grid(row=6, column=0, columnspan=2)

    Labeling_Padding = tk.Label(LabelingMainFrame, text="", font=("Arial", 20)).grid(row=4)
    Label_CheckLabel = tk.Label(LabelingMainFrame, text="Number of files to visually inspect:").grid(row=5, column=0)
    global Label_CheckNumber
    Label_CheckNumber = StringVar()
    Label_CheckNumber.set("5")
    Labeling_SpectrogramCheckNumber = tk.Spinbox(LabelingMainFrame,from_=1, to=99, textvariable=Label_CheckNumber, width=15)
    Labeling_SpectrogramCheckNumber.grid(row=5, column=1, sticky="W",padx=Padding_Width)

    Labeling_Padding2 = tk.Label(LabelingMainFrame, text="", font=("Arial", 20)).grid(row=7)

    LabelingButton = tk.Button(LabelingMainFrame, text="Run", command = lambda : Labeling())
    LabelingButton.grid(row=8, column=0, columnspan=2)

    global LabelingBirdIDText
    LabelingBirdIDText = tk.Entry(LabelingMainFrame, font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
    LabelingBirdIDText.insert(0, "Bird ID")
    LabelingBirdIDText.bind("<FocusIn>", focus_in)
    LabelingBirdIDText.bind("<FocusOut>", focus_out)
    LabelingBirdIDText.grid(row=2, column=1, sticky="w", padx=Padding_Width)

    LabelingBirdIDLabel = tk.Label(LabelingMainFrame, text="Bird ID:")
    LabelingBirdIDLabel.grid(row=2, column=0)

    global LabelingDirectoryText
    LabelingDirectoryText = tk.Label(LabelingMainFrame, text=Dir_Width*" ", bg="light grey", anchor="w")
    LabelingDirectoryText.grid(row=1, column=1, columnspan=2, sticky="w", padx=Padding_Width)
    LabelingFileExplorer = tk.Button(LabelingMainFrame, text="Find Segmentation File", command=lambda: FileExplorer("Labeling", "Input"))
    LabelingFileExplorer.grid(row=1, column=0)

    global LabelingOutputFile_Text
    # global LabelingOutputFile_Var
    # LabelingOutputFile_Var = tk.StringVar()
    LabelingOutputFile_Text = tk.Label(LabelingMainFrame, text=Dir_Width*" ", bg="light grey", anchor="w")
    # LabelingOutputFile_Var.set("")
    LabelingOutputFile_Text.grid(row=3, column=1, columnspan=2, sticky="w", padx=Padding_Width)

    def LabelingOutputDirectory():
        global LabelingOutputFile_Text
        Selected_Output = filedialog.askdirectory()
        LabelingOutputFile_Text.config(text=str(Selected_Output))
        LabelingOutputFile_Text.update()

    LabelingOutputFile_Button = tk.Button(LabelingMainFrame, text="Find Output Folder", command=lambda:FileExplorer("Labeling", "Output"))
    LabelingOutputFile_Button.grid(row=3, column=0)

    # Labeling Spectrogram and UMAP Generation
    global LabelingFileDisplay
    LabelingFileDisplay = tk.Label(LabelingSpectrogram_UMAP, text=Dir_Width*" ", bg="light grey")
    LabelingFileDisplay.grid(row=0, column=1, sticky="w", padx=Padding_Width)
    FindLabelingFile_Button = tk.Button(LabelingSpectrogram_UMAP, text="Select Labeling File", command=lambda:FileExplorer("Labeling_UMAP", "Input"))
    FindLabelingFile_Button.grid(row=0, column=0)

    BirdIDText_Labeling = tk.Label(LabelingSpectrogram_UMAP, text="Bird ID")
    BirdIDText_Labeling.grid(row=1, column=0)
    global BirdID_Labeling
    BirdID_Labeling = tk.Entry(LabelingSpectrogram_UMAP,font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
    BirdID_Labeling.insert(0, "Bird ID")
    BirdID_Labeling.bind("<FocusIn>", focus_in)
    BirdID_Labeling.bind("<FocusOut>", focus_out)
    BirdID_Labeling.grid(row=1, column=1, sticky="w", padx=Padding_Width)
    LabelingUMAP_OutputButton = tk.Button(LabelingSpectrogram_UMAP, text="Select Output Folder", command=lambda:FileExplorer("Labeling_UMAP", "Output"))
    LabelingUMAP_OutputButton.grid(row=2, column=0)
    global LabelingUMAP_OutputText
    LabelingUMAP_OutputText = tk.Label(LabelingSpectrogram_UMAP, text=Dir_Width*" ", bg="light grey")
    LabelingUMAP_OutputText.grid(row=2, column=1, sticky="w", padx=Padding_Width)

    global MaxFiles_Labeling
    MaxFiles_Labeling_Text = tk.Label(LabelingSpectrogram_UMAP, text="Number of Spectrograms to Generate:").grid(row=3, column=0)
    MaxFiles_Labeling = tk.Spinbox(LabelingSpectrogram_UMAP, from_=0, to=99, font=("Arial, 10"), width=10, justify="center")
    MaxFiles_Labeling.grid(row=3, column=1, sticky="N")

    GenerateSpectrogram_Button = tk.Button(LabelingSpectrogram_UMAP, text="Generate Spectrograms", command=lambda:LabelingSpectrograms())
    GenerateSpectrogram_Button.grid(row=4, column=0)

    # Labeling Bulk Save
    LabelingBirdIDText2 = tk.Entry(LabelingBulkSave, font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
    LabelingBirdIDText2.insert(0, "Bird ID")
    LabelingBirdIDText2.bind("<FocusIn>", focus_in)
    LabelingBirdIDText2.bind("<FocusOut>", focus_out)
    LabelingBirdIDText2.grid(row=2, column=1, sticky="w", padx=Padding_Width)

    LabelingBirdIDLabel2 = tk.Label(LabelingBulkSave, text="Bird ID:")
    LabelingBirdIDLabel2.grid(row=2, column=0)
    LabelingFileExplorer2 = tk.Button(LabelingBulkSave, text="Find Labeling File",command=lambda: LabelingFileExplorerFunctionSaving())
    LabelingFileExplorer2.grid(row=0, column=0)

    LabelingErrorMessage = tk.Label(LabelingMainFrame, text="")
    LabelingErrorMessage.grid(row=10, column=1, padx=Padding_Width)

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
    LabelingSaveAll.grid(row=4, column=1, padx=Padding_Width)

    LabelingBulkSaveButton = tk.Button(LabelingBulkSave, text="Save", command=lambda:LabelingSavePhotos())
    LabelingBulkSaveButton.grid(row=5, column=0, columnspan=2)

    LabelingBulkSaveError = tk.Label(LabelingBulkSave, text="")
    LabelingBulkSaveError.grid(row=6, column=0, columnspan=2)

    def BulkLabelingFileText():
        global BulkLabelFiles_to_save
        BulkLabelFiles_to_save = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
        BulkLabelFiles = tk.Label(LabelingBulkSave, text=str(len(BulkLabelFiles_to_save)) + " files selected")
        BulkLabelFiles.grid(row=1, column=1)

    LabelingBulkFileSelect = tk.Button(LabelingBulkSave, text="Select Song Files to Save", command=lambda:BulkLabelingFileText())
    LabelingBulkFileSelect.grid(row=1, column=0)

    ### Labeling Spectrogram Settings ###

    def ResetLabelingSetting(Variable, EntryList, ErrorLabel=1):
        DefaultValues = [500, 15000, 0.00001, 2, -28, 512, 512, 128, 300]
        Entries = [bandpass_lower_cutoff_entry_Labeling,bandpass_upper_cutoff_entry_Labeling,a_min_entry_Labeling,ref_db_entry_Labeling,
                   min_level_db_entry_Labeling,n_fft_entry_Labeling,win_length_entry_Labeling,hop_length_entry_Labeling,max_spec_size_entry_Labeling]
        if Variable == "all":
            global LabelErrorLabels
            for Label in LabelErrorLabels:
                Label.config(text="")
            i = -1
            for entry in Entries:
                i+=1
                entry.delete(0, END)
                entry.insert(0, str(DefaultValues[i]))
                entry.config(bg="white")
                entry.update()
        else:
            Variable.delete(0, END)
            Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
            Variable.update()
            Variable.config(bg="white")
            ErrorLabel.config(text="")

    Dialog_TitleWidth = 20
    Dialog_TitleHeight = 2
    Dialog_BodyWidth = 50

    LabelingSpectrogramDialog = tk.Label(LabelingSettingsFrame, text="", justify="center", width=Dialog_BodyWidth)
    LabelingSpectrogramDialog.grid(row=3, column=4, rowspan=10)
    LabelingSpectrogramTitle = tk.Label(LabelingSettingsFrame, text="", justify="center", font=("Arial", 20, 'bold'), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
    LabelingSpectrogramTitle.grid(row=0, column=4, rowspan= 3)

    LabelingUMAPDialog = tk.Label(LabelingUMAPSettings, text="", justify="center")
    LabelingUMAPDialog.grid(row=1, column=4, rowspan=6)
    LabelingUMAPTitle = tk.Label(LabelingUMAPSettings, text="", justify="center", font=("Arial", 20, 'bold'), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
    LabelingUMAPTitle.grid(row=0, column=4)

    ErrorFont = ("Arial", 7)

    bandpass_lower_cutoff_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    bandpass_lower_cutoff_Labeling_Error.grid(row=1, column=1)
    bandpass_lower_cutoff_text_Labeling = tk.Label(LabelingSettingsFrame, text="bandpass_lower_cutoff").grid(row=0, column=0)
    bandpass_lower_cutoff_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    bandpass_lower_cutoff_entry_Labeling.insert(0, "500")
    bandpass_lower_cutoff_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(bandpass_lower_cutoff_entry_Labeling, "bandpass_lower_cutoff_entry_Labeling",bandpass_lower_cutoff_Labeling_Error))
    bandpass_lower_cutoff_entry_Labeling.grid(row=0, column=1)
    bandpass_lower_cutoff_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(bandpass_lower_cutoff_entry_Labeling, LabelSpecList,bandpass_lower_cutoff_Labeling_Error)).grid(row=0, column=2)
    bandpass_lower_cutoff_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    bandpass_lower_cutoff_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    bandpass_lower_cutoff_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    bandpass_lower_cutoff_moreinfo_Labeling.bind("<Leave>", LessInfo)

    bandpass_upper_cutoff_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    bandpass_upper_cutoff_Labeling_Error.grid(row=3, column=1)
    bandpass_upper_cutoff_text_Labeling = tk.Label(LabelingSettingsFrame, text="bandpass_upper_cutoff").grid(row=2,column=0)
    bandpass_upper_cutoff_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    bandpass_upper_cutoff_entry_Labeling.insert(0, "15000")
    bandpass_upper_cutoff_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(bandpass_upper_cutoff_entry_Labeling,"bandpass_upper_cutoff_entry_Labeling",bandpass_upper_cutoff_Labeling_Error))
    bandpass_upper_cutoff_entry_Labeling.grid(row=2, column=1)
    bandpass_upper_cutoff_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(bandpass_upper_cutoff_entry_Labeling, LabelSpecList,bandpass_upper_cutoff_Labeling_Error)).grid(row=2, column=2)
    bandpass_upper_cutoff_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    bandpass_upper_cutoff_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
    bandpass_upper_cutoff_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    bandpass_upper_cutoff_moreinfo_Labeling.bind("<Leave>", LessInfo)

    a_min_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    a_min_Labeling_Error.grid(row=5, column=1)
    a_min_text_Labeling = tk.Label(LabelingSettingsFrame, text="a_min").grid(row=4,column=0)
    a_min_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    a_min_entry_Labeling.insert(0, "0.00001")
    a_min_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(a_min_entry_Labeling,"a_min_entry_Labeling",a_min_Labeling_Error))
    a_min_entry_Labeling.grid(row=4, column=1)
    a_min_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(a_min_entry_Labeling, LabelSpecList,a_min_Labeling_Error)).grid(row=4, column=2)
    a_min_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    a_min_moreinfo_Labeling.grid(row=4, column=3, sticky=W)
    a_min_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    a_min_moreinfo_Labeling.bind("<Leave>", LessInfo)

    ref_db_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    ref_db_Labeling_Error.grid(row=7, column=1)
    ref_db_text_Labeling = tk.Label(LabelingSettingsFrame, text="ref_db").grid(row=6,column=0)
    ref_db_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    ref_db_entry_Labeling.insert(0, "20")
    ref_db_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(ref_db_entry_Labeling,"ref_db_entry_Labeling",ref_db_Labeling_Error))
    ref_db_entry_Labeling.grid(row=6, column=1)
    ref_db_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(ref_db_entry_Labeling, LabelSpecList,ref_db_Labeling_Error)).grid(row=6, column=2)
    ref_db_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    ref_db_moreinfo_Labeling.grid(row=6, column=3, sticky=W)
    ref_db_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    ref_db_moreinfo_Labeling.bind("<Leave>", LessInfo)

    min_level_db_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    min_level_db_Labeling_Error.grid(row=9, column=1)
    min_level_db_text_Labeling = tk.Label(LabelingSettingsFrame, text="min_level_db").grid(row=8,column=0)
    min_level_db_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    min_level_db_entry_Labeling.insert(0, "-28")
    min_level_db_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(min_level_db_entry_Labeling,'min_level_db_entry_Labeling',min_level_db_Labeling_Error))
    min_level_db_entry_Labeling.grid(row=8, column=1)
    min_level_db_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(min_level_db_entry_Labeling, LabelSpecList,min_level_db_Labeling_Error)).grid(row=8, column=2)
    min_level_db_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    min_level_db_moreinfo_Labeling.grid(row=8, column=3, sticky=W)
    min_level_db_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_level_db_moreinfo_Labeling.bind("<Leave>", LessInfo)

    n_fft_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    n_fft_Labeling_Error.grid(row=11, column=1)
    n_fft_text_Labeling = tk.Label(LabelingSettingsFrame, text="n_fft").grid(row=10,column=0)
    n_fft_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    n_fft_entry_Labeling.insert(0, "512")
    n_fft_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(n_fft_entry_Labeling,'n_fft_entry_Labeling',n_fft_Labeling_Error))
    n_fft_entry_Labeling.grid(row=10, column=1)
    n_fft_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(n_fft_entry_Labeling, LabelSpecList,n_fft_Labeling_Error)).grid(row=10, column=2)
    n_fft_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    n_fft_moreinfo_Labeling.grid(row=10, column=3, sticky=W)
    n_fft_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_fft_moreinfo_Labeling.bind("<Leave>", LessInfo)

    win_length_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    win_length_Labeling_Error.grid(row=13, column=1)
    win_length_text_Labeling = tk.Label(LabelingSettingsFrame, text="win_length").grid(row=12,column=0)
    win_length_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    win_length_entry_Labeling.insert(0, "512")
    win_length_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(win_length_entry_Labeling,'win_length_entry_Labeling',win_length_Labeling_Error))
    win_length_entry_Labeling.grid(row=12, column=1)
    win_length_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(win_length_entry_Labeling, LabelSpecList,win_length_Labeling_Error)).grid(row=12, column=2)
    win_length_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    win_length_moreinfo_Labeling.grid(row=12, column=3, sticky=W)
    win_length_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    win_length_moreinfo_Labeling.bind("<Leave>", LessInfo)

    hop_length_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    hop_length_Labeling_Error.grid(row=15, column=1)
    hop_length_text_Labeling = tk.Label(LabelingSettingsFrame, text="hop_length").grid(row=14,column=0)
    hop_length_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    hop_length_entry_Labeling.insert(0, "128")
    hop_length_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(hop_length_entry_Labeling,'hop_length_entry_Labeling',hop_length_Labeling_Error))
    hop_length_entry_Labeling.grid(row=14, column=1)
    hop_length_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(hop_length_entry_Labeling, LabelSpecList,hop_length_Labeling_Error)).grid(row=14, column=2)
    hop_length_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    hop_length_moreinfo_Labeling.grid(row=14, column=3, sticky=W)
    hop_length_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    hop_length_moreinfo_Labeling.bind("<Leave>", LessInfo)

    max_spec_size_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
    max_spec_size_Labeling_Error.grid(row=17, column=1)
    max_spec_size_text_Labeling = tk.Label(LabelingSettingsFrame, text="max_spec_size").grid(row=16,column=0)
    max_spec_size_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
    max_spec_size_entry_Labeling.insert(0, "300")
    max_spec_size_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(max_spec_size_entry_Labeling,'max_spec_size_entry_Labeling',max_spec_size_Labeling_Error))
    max_spec_size_entry_Labeling.grid(row=16, column=1)
    max_spec_size_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(max_spec_size_entry_Labeling, LabelSpecList,max_spec_size_Labeling_Error)).grid(row=16, column=2)
    max_spec_size_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
    max_spec_size_moreinfo_Labeling.grid(row=16, column=3, sticky=W)
    max_spec_size_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    max_spec_size_moreinfo_Labeling.bind("<Leave>", LessInfo)

    LabelSpecList = [bandpass_lower_cutoff_entry_Labeling,bandpass_upper_cutoff_entry_Labeling,a_min_entry_Labeling,ref_db_entry_Labeling,
                     min_level_db_entry_Labeling,n_fft_entry_Labeling,win_length_entry_Labeling,hop_length_entry_Labeling,max_spec_size_entry_Labeling]
    LabelingSpectrogram_ResetAllSettings = tk.Button(LabelingSettingsFrame, text="Reset All", command=lambda:ResetLabelingSetting("all", LabelSpecList))
    LabelingSpectrogram_ResetAllSettings.grid(row=25, column=1)

    global LabelErrorLabels
    LabelErrorLabels = [bandpass_lower_cutoff_Labeling_Error,bandpass_upper_cutoff_Labeling_Error,a_min_Labeling_Error,ref_db_Labeling_Error,min_level_db_Labeling_Error,n_fft_Labeling_Error,win_length_Labeling_Error,hop_length_Labeling_Error,max_spec_size_Labeling_Error]

    LabelingBulkSaveError = tk.Label(LabelingBulkSave, textvariable="")
    LabelingBulkSaveError.grid()
    ### Labeling UMAP Settings ###

    def ResetLabelingUMAPSetting(Variable, EntryList, ErrorLabel):
        DefaultValues = [10, 2, 0.0, 1, "euclidean", "None"]
        Entries = [n_neighbors_entry_Labeling,n_components_entry_Labeling,min_dist_entry_Labeling,spread_entry_Labeling,metric_variable,random_state_entry_Labeling]
        global LabelingUMAPErrorLabels
        if Variable == "all":
            global LabelingUMAPErrorLabels
            for Label in LabelingUMAPErrorLabels:
                Label.config(text="")
                Label.update()
            i = -1
            for entry in Entries:
                i+=1
                if i == 4:
                    metric_variable.set(DefaultValues[4])
                    metric_entry_Labeling.update()
                else:
                    entry.delete(0, END)
                    entry.insert(0, str(DefaultValues[i]))
                    entry.config(bg="white")
                    entry.update()
            for error in LabelingUMAPErrorLabels:
                error.config(text="")
        else:
            if Variable != metric_entry_Labeling:
                Variable.delete(0, END)
                Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
                Variable.config(bg="white")
                Variable.update()
                ErrorLabel.config(text="")
            else:
                metric_variable.set(DefaultValues[4])
                metric_entry_Labeling.update()

    n_neighbors_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    n_neighbors_Labeling_Error.grid(row=1, column=1)
    n_neighbors_text_Labeling = tk.Label(LabelingUMAPSettings, text="n_neighbors").grid(row=0, column=0)
    n_neighbors_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    n_neighbors_entry_Labeling.insert(0, "10")
    n_neighbors_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(n_neighbors_entry_Labeling,'n_neighbors_entry_Labeling',n_neighbors_Labeling_Error))
    n_neighbors_entry_Labeling.grid(row=0, column=1)
    n_neighbors_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_neighbors_entry_Labeling, LabelingUMAPVars,n_neighbors_Labeling_Error)).grid(row=0, column=2)
    n_neighbors_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    n_neighbors_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    n_neighbors_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_neighbors_moreinfo_Labeling.bind("<Leave>", LessInfo)

    n_components_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    n_components_Labeling_Error.grid(row=3, column=1)
    n_components_text_Labeling = tk.Label(LabelingUMAPSettings, text="n_components").grid(row=2, column=0)
    n_components_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    n_components_entry_Labeling.insert(0, "2")
    n_components_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(n_components_entry_Labeling,'n_components_entry_Labeling',n_components_Labeling_Error))
    n_components_entry_Labeling.grid(row=2, column=1)
    n_components_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(n_components_entry_Labeling, LabelingUMAPVars,n_components_Labeling_Error)).grid(row=2, column=2)
    n_components_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    n_components_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
    n_components_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    n_components_moreinfo_Labeling.bind("<Leave>", LessInfo)

    min_dist_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    min_dist_Labeling_Error.grid(row=5, column=1)
    min_dist_text_Labeling = tk.Label(LabelingUMAPSettings, text="min_dist").grid(row=4, column=0)
    min_dist_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    min_dist_entry_Labeling.insert(0, "0.0")
    min_dist_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(min_dist_entry_Labeling,'min_dist_entry_Labeling',min_dist_Labeling_Error))
    min_dist_entry_Labeling.grid(row=4, column=1)
    min_dist_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(min_dist_entry_Labeling, LabelingUMAPVars,min_dist_Labeling_Error)).grid(row=4, column=2)
    min_dist_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    min_dist_moreinfo_Labeling.grid(row=4, column=3, sticky=W)
    min_dist_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_dist_moreinfo_Labeling.bind("<Leave>", LessInfo)

    spread_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    spread_Labeling_Error.grid(row=7, column=1)
    spread_text_Labeling = tk.Label(LabelingUMAPSettings, text="spread").grid(row=6, column=0)
    spread_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    spread_entry_Labeling.insert(0, "1.0")
    spread_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(spread_entry_Labeling,'spread_entry_Labeling',spread_Labeling_Error))
    spread_entry_Labeling.grid(row=6, column=1)
    spread_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(spread_entry_Labeling, LabelingUMAPVars,spread_Labeling_Error)).grid(row=6, column=2)
    spread_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    spread_moreinfo_Labeling.grid(row=6, column=3, sticky=W)
    spread_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    spread_moreinfo_Labeling.bind("<Leave>", LessInfo)

    metric_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    metric_Labeling_Error.grid(row=9, column=1)
    metric_text_Labeling = tk.Label(LabelingUMAPSettings, text="metric").grid(row=8, column=0)
    options_list = ['euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','mahalanobis','wminkowski','seuclidean','cosine','correlation','haversine','hamming','jaccard','dice','russelrao','kulsinski','ll_dirichlet','hellinger','rogerstanimoto','sokalmichener','sokalsneath','yule']
    metric_variable = StringVar()
    metric_variable.set(options_list[0])
    metric_entry_Labeling = tk.OptionMenu(LabelingUMAPSettings, metric_variable, *options_list)
    metric_entry_Labeling.grid(row=8, column=1)
    metric_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(metric_entry_Labeling, LabelingUMAPVars,metric_Labeling_Error)).grid(row=8, column=2)
    metric_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    metric_moreinfo_Labeling.grid(row=8, column=3, sticky=W)
    metric_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    metric_moreinfo_Labeling.bind("<Leave>", LessInfo)

    random_state_Labeling_Error = tk.Label(LabelingUMAPSettings, text="", font=ErrorFont)
    random_state_Labeling_Error.grid(row=11, column=1)
    random_state_text_Labeling = tk.Label(LabelingUMAPSettings, text="random_state").grid(row=10, column=0)
    random_state_entry_Labeling = tk.Entry(LabelingUMAPSettings, justify="center")
    random_state_entry_Labeling.insert(0, "None")
    random_state_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(random_state_entry_Labeling,'random_state_entry_Labeling',random_state_Labeling_Error))
    random_state_entry_Labeling.grid(row=10, column=1)
    random_state_reset_Labeling = tk.Button(LabelingUMAPSettings, text="Reset",command=lambda: ResetLabelingUMAPSetting(random_state_entry_Labeling, LabelingUMAPVars,random_state_Labeling_Error)).grid(row=10, column=2)
    random_state_moreinfo_Labeling = tk.Button(LabelingUMAPSettings, text='?', state="disabled", fg="black")
    random_state_moreinfo_Labeling.grid(row=10, column=3, sticky=W)
    random_state_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    random_state_moreinfo_Labeling.bind("<Leave>", LessInfo)

    global LabelingUMAPErrorLabels
    LabelingUMAPErrorLabels = [n_neighbors_Labeling_Error,n_components_Labeling_Error,min_dist_Labeling_Error,spread_Labeling_Error,metric_Labeling_Error,random_state_Labeling_Error]

    LabelingUMAPVars = [n_neighbors_entry_Labeling,n_components_entry_Labeling,min_dist_entry_Labeling,spread_entry_Labeling,metric_entry_Labeling,random_state_entry_Labeling]
    LabelingUMAPSettingsResetAll = tk.Button(LabelingUMAPSettings, text="Reset All", command=lambda:ResetLabelingUMAPSetting("all", LabelingUMAPVars, "all"))
    LabelingUMAPSettingsResetAll.grid(row=25, column=1)

    ### Labeling Clustering Settings ###
    def ResetLabelingClusterSetting(Variable, EntryList, ErrorLabel):
        DefaultValues = [0.04, 5]
        if Variable == "all":
            i = -1
            for entry in Entries:
                i+=1
                entry.delete(0, END)
                entry.insert(0, str(DefaultValues[i]))
                entry.config(bg="white")
                entry.update()
            min_cluster_prop_Labeling_Error.config(text="")
            min_samples_Labeling_Error.config(text="")
        else:
            Variable.delete(0, END)
            Variable.insert(0, str(DefaultValues[EntryList.index(Variable)]))
            Variable.config(bg="white")
            Variable.update()
            ErrorLabel.config(text="")

    min_cluster_prop_Labeling_Error = tk.Label(LabelingClusterSettings, text="", font=ErrorFont)
    min_cluster_prop_Labeling_Error.grid(row=1, column=1)
    min_cluster_prop_text_Labeling = tk.Label(LabelingClusterSettings, text="min_cluster_prop").grid(row=0, column=0)
    min_cluster_prop_entry_Labeling = tk.Entry(LabelingClusterSettings, justify="center")
    min_cluster_prop_entry_Labeling.insert(0, "0.04")
    min_cluster_prop_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(min_cluster_prop_entry_Labeling,'min_cluster_prop_entry_Labeling',min_cluster_prop_Labeling_Error))
    min_cluster_prop_entry_Labeling.grid(row=0, column=1)
    min_cluster_prop_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(min_cluster_prop_entry_Labeling,LabelingClusterVars,min_cluster_prop_Labeling_Error)).grid(row=0, column=2)
    min_cluster_prop_moreinfo_Labeling = tk.Button(LabelingClusterSettings, text='?', state="disabled", fg="black")
    min_cluster_prop_moreinfo_Labeling.grid(row=0, column=3, sticky=W)
    min_cluster_prop_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_cluster_prop_moreinfo_Labeling.bind("<Leave>", LessInfo)

    min_samples_Labeling_Error = tk.Label(LabelingClusterSettings, text="", font=ErrorFont)
    min_samples_Labeling_Error.grid(row=3, column=1)
    min_samples_text_Labeling = tk.Label(LabelingClusterSettings, text="min_samples").grid(row=2, column=0)
    min_samples_entry_Labeling = tk.Entry(LabelingClusterSettings, justify="center")
    min_samples_entry_Labeling.insert(0, "5")
    min_samples_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(min_samples_entry_Labeling,'min_samples_entry_Labeling',min_samples_Labeling_Error))
    min_samples_entry_Labeling.grid(row=2, column=1)
    min_samples_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(min_samples_entry_Labeling,LabelingClusterVars,min_samples_Labeling_Error)).grid(row=2, column=2)
    min_samples_moreinfo_Labeling = tk.Button(LabelingClusterSettings, text='?', state="disabled", fg="black")
    min_samples_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
    min_samples_moreinfo_Labeling.bind("<Enter>", MoreInfo)
    min_samples_moreinfo_Labeling.bind("<Leave>", LessInfo)

    LabelingClusterVars = [min_cluster_prop_entry_Labeling,spread_entry_Labeling]
    LabelingClusterResetAll = tk.Button(LabelingClusterSettings, text="Reset All", command=lambda: ResetLabelingClusterSetting("all", LabelingClusterVars, "all"))
    LabelingClusterResetAll.grid(row=4, column=1)

    LabelingClusterDialog = tk.Label(LabelingClusterSettings, text="")
    LabelingClusterDialog.grid(row=1, column=4, rowspan=10)
    LabelingClusterTitle = tk.Label(LabelingClusterSettings, text="", font=("Arial", 20, "bold"), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
    LabelingClusterTitle.grid(row=0, column=4)

    LabelingClusterPadding = tk.Label(LabelingClusterSettings, pady=10, font=("Arial",25))
    LabelingClusterPadding.grid(row=5, column=1)

    LabelingClusterVars = [min_cluster_prop_entry_Labeling,min_samples_entry_Labeling]

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
    LoadLabelingSettings.grid(row=25, column=0)

    ### Single File Acoustic Features Window ###

    AcousticsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(AcousticsFrame, text="Acoustic Features")

    AcousticsMainFrameSingle = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook = ttk.Notebook(AcousticsFrame)
    AcousticsNotebook.grid(row=1)
    AcousticsNotebook.add(AcousticsMainFrameSingle, text="Single Song Interval")

    AcousticsInputLabel = tk.Label(AcousticsMainFrameSingle, text="Input Type:").grid(row=0, column=0)
    global AcousticsInputVar
    AcousticsInputVar = IntVar()
    AcousticsInputType_Single = tk.Radiobutton(AcousticsMainFrameSingle, text="Single File", variable=AcousticsInputVar, value=0).grid(row=0, column=1)
    AcousticsInputType_Folder = tk.Radiobutton(AcousticsMainFrameSingle, text="Whole Folder", variable=AcousticsInputVar, value=1).grid(row=0, column=2)


    AcousticsFileExplorer = tk.Button(AcousticsMainFrameSingle, text="Select Input",
                                      command=lambda: FileExplorer("Acoustics_Single", "Input"))
    AcousticsFileExplorer.grid(row=1, column=0)
    AcousticsFileDisplay = tk.Label(AcousticsMainFrameSingle, text=2*Dir_Width*" ", bg="light grey", anchor=tk.W)
    AcousticsFileDisplay.grid(row=1, column=1, padx=Padding_Width, columnspan=2)

    AcousticsOutput_Button = tk.Button(AcousticsMainFrameSingle, text="Select Output Folder",
                                       command=lambda: FileExplorer("Acoustics_Single", "Output"))
    AcousticsOutput_Button.grid(row=2, column=0)
    global AcousticsOutput_Text
    AcousticsOutput_Text = tk.Label(AcousticsMainFrameSingle, text=2*Dir_Width*" ", bg="light grey")
    AcousticsOutput_Text.grid(row=2, column=1, padx=Padding_Width, columnspan=2)

    Calculation_Method_Label = tk.Label(AcousticsMainFrameSingle, text="Calculation Method:").grid(row=3, column=0)
    global ContCalc
    ContCalc = IntVar()
    ContCalc.set(0)
    Continuous_Calculations = tk.Radiobutton(AcousticsMainFrameSingle, text="Continuous Calcualtions",
                                             variable=ContCalc, value=0).grid(row=3, column=1, columnspan=1)
    Average_Calculations = tk.Radiobutton(AcousticsMainFrameSingle, text="Average Calcualtions", variable=ContCalc,
                                          value=1).grid(row=3, column=2, columnspan=1)

    PaddingLabel = tk.Label(AcousticsMainFrameSingle, text="").grid(row=4)

    AcousticsText = tk.Label(AcousticsMainFrameSingle, text="Acoustic features to analyze:", justify="center")
    AcousticsText.grid(row=5, column=1, padx=Padding_Width, columnspan=2)

    AcousticsOnsetLabel = tk.Label(AcousticsMainFrameSingle, text="Onset:").grid(row=5, column=0)
    AcousticsOnset = tk.Entry(AcousticsMainFrameSingle, justify="center")
    AcousticsOnset.insert(0,"0")
    AcousticsOnset.grid(row=6, column=0)

    AcousticsOffsetLabel = tk.Label(AcousticsMainFrameSingle, text="Offset:").grid(row=8, column=0)
    global AcousticsOffset
    AcousticsOffset = tk.Entry(AcousticsMainFrameSingle, justify="center")
    AcousticsOffset.insert(0, "End of File")
    AcousticsOffset.grid(row=9, column=0)

    AcousticsOffsetReset = tk.Button(AcousticsMainFrameSingle, text="Reset", command=lambda:ResetAcousticsOffset)
    AcousticsOffsetReset.grid(row=10, column=0)


    # global SaveMetadata
    # SaveMetadata = IntVar()
    # SaveMetadata_Checkbox = tk.Checkbutton(AcousticsMainFrameSingle, text="Save Metadata", variable=SaveMetadata, onvalue=1, offvalue=0).grid

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

    def CheckAllBoxes(CheckAll, Mode):
        if Mode == "Interval":
            global RunGoodness
            global RunMean_frequency
            global RunEntropy
            global RunAmplitude
            global RunAmplitude_modulation
            global RunFrequency_modulation
            global RunPitch
            ButtonNameList = [RunGoodness,RunMean_frequency,RunEntropy,RunAmplitude,RunAmplitude_modulation,RunFrequency_modulation,RunPitch]
        if Mode == "Multi":
            global MultiRunGoodness
            global MultiRunMean_frequency
            global MultiRunEntropy
            global MultiRunAmplitude
            global MultiRunAmplitude_modulation
            global MultiRunFrequency_modulation
            global MultiRunPitch
            ButtonNameList = [MultiRunGoodness, MultiRunMean_frequency, MultiRunEntropy, MultiRunAmplitude,
                                   MultiRunAmplitude_modulation,
                                   MultiRunFrequency_modulation, MultiRunPitch]
        if Mode == "RunAll":
            global RunAll_RunGoodness
            global RunAll_RunMean_frequency
            global RunAll_RunEntropy
            global RunAll_RunAmplitude
            global RunAll_RunAmplitude_modulation
            global RunAll_RunFrequency_modulation
            global RunAll_RunPitch
            ButtonNameList = [RunAll_RunGoodness, RunAll_RunMean_frequency, RunAll_RunEntropy, RunAll_RunAmplitude, RunAll_RunAmplitude_modulation,
                              RunAll_RunFrequency_modulation, RunAll_RunPitch]
        if CheckAll.get() == 1:
            for checkbox in ButtonNameList:
                checkbox.set(1)
        if CheckAll.get() == 0:
            for checkbox in ButtonNameList:
                checkbox.set(0)

    StartRow = 6
    CheckAll_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text='Select All', variable=CheckAll, command=lambda:CheckAllBoxes(CheckAll,"Interval"))
    CheckAll_CheckBox.grid(row=StartRow, column=1, columnspan=2)
    Goodness_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text= "Goodness", anchor=tk.W, variable=RunGoodness)
    Goodness_CheckBox.grid(row=StartRow+1, column=1, columnspan=2)
    Mean_frequency_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Mean Frequency",anchor=tk.W, variable=RunMean_frequency)
    Mean_frequency_CheckBox.grid(row=StartRow+2, column=1, columnspan=2)
    Entropy_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Entropy", anchor=tk.W, variable=RunEntropy)
    Entropy_CheckBox.grid(row=StartRow+3, column=1, columnspan=2)
    Amplitude_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Amplitude", anchor=tk.W, variable=RunAmplitude)
    Amplitude_CheckBox.grid(row=StartRow+4, column=1, columnspan=2)
    Amplitude_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Amplitude Modulation", anchor=tk.W, variable=RunAmplitude_modulation)
    Amplitude_modulation_CheckBox.grid(row=StartRow+5, column=1, columnspan=2)
    Frequency_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Frequency Modulation", anchor=tk.W, variable=RunFrequency_modulation)
    Frequency_modulation_CheckBox.grid(row=StartRow+6, column=1, columnspan=2)
    Pitch_CheckBox = tk.Checkbutton(AcousticsMainFrameSingle, text="Pitch", anchor=tk.W, variable=RunPitch)
    Pitch_CheckBox.grid(row=StartRow+7, column=1, columnspan=2)

    PaddingLabel2 = tk.Label(AcousticsMainFrameSingle, text="").grid(row=StartRow+8)

    AcousticsRunButton = tk.Button(AcousticsMainFrameSingle, text="Run", command=lambda: Acoustics_Interval())
    AcousticsRunButton.grid(row=StartRow+10, column=1, columnspan=2)

    ### Multiple Syllable Acoustic Features Window ###

    AcousticsMainFrameMulti = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsMainFrameMulti, text="Multiple Syllables")
    AcousticsSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    AcousticsNotebook.add(AcousticsSettingsFrame, text="Advanced Settings")
    # AcousticsInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    # AcousticsNotebook.add(AcousticsInfoFrame, text="Info")

    MultiAcousticsError = tk.Label(AcousticsMainFrameMulti, text="")
    MultiAcousticsError.grid(row=11, column=1)


    ### Advanced Acoustics Settings ###

    AcousticsSettingsDialogTitle = tk.Label(AcousticsSettingsFrame, text="", justify="center", font=("Arial", 25, "bold"), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
    AcousticsSettingsDialogTitle.grid(row=0, column=4)
    AcousticSettingsDialog = tk.Label(AcousticsSettingsFrame, text="", justify="center")
    AcousticSettingsDialog.grid(row=1, column=4, rowspan=8)

    def ResetAcousticSetting(Variable, EntryList, ErrorLabel):
        DefaultValues = [400,40,1024,1830,380,0.5,70, 8000]
        global AcousticErrorLabels
        if Variable == "all":
            i = -1
            for var in EntryList:
                i+=1
                var.delete(0, END)
                var.insert(0, str(DefaultValues[i]))
                var.config(bg="white")
                var.update()
            for label in AcousticErrorLabels:
                label.config(text="")

        else:
            Variable.delete(0, END)
            Variable.insert(0,str(DefaultValues[EntryList.index(Variable)]))
            Variable.config(bg="white")
            Variable.update()
            ErrorLabel.config(text="")

    win_length_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    win_length_Acoustics_Error.grid(row=1, column=1)
    win_length_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="win_length").grid(row=0, column=0)
    win_length_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    win_length_entry_Acoustics.insert(0, "400")
    win_length_entry_Acoustics.grid(row=0, column=1)
    win_length_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(win_length_entry_Acoustics, "win_length_entry_Acoustics",win_length_Acoustics_Error))
    win_length_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(win_length_entry_Acoustics, EntryList,win_length_Acoustics_Error)).grid(row=0, column=2)
    win_length_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    win_length_moreinfo_Acoustics.grid(row=0, column=3, sticky=W)
    win_length_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    win_length_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    hop_length_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    hop_length_Acoustics_Error.grid(row=3, column=1)
    hop_length_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="hop_length").grid(row=2, column=0)
    hop_length_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    hop_length_entry_Acoustics.insert(0, "40")
    hop_length_entry_Acoustics.grid(row=2, column=1)
    hop_length_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(hop_length_entry_Acoustics, "hop_length_entry_Acoustics",hop_length_Acoustics_Error))
    hop_length_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(hop_length_entry_Acoustics, EntryList,hop_length_Acoustics_Error)).grid(row=2, column=2)
    hop_length_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    hop_length_moreinfo_Acoustics.grid(row=2, column=3, sticky=W)
    hop_length_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    hop_length_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    n_fft_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    n_fft_Acoustics_Error.grid(row=5, column=1)
    n_fft_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="n_fft").grid(row=4, column=0)
    n_fft_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    n_fft_entry_Acoustics.insert(0, "1024")
    n_fft_entry_Acoustics.grid(row=4, column=1)
    n_fft_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(n_fft_entry_Acoustics, "n_fft_entry_Acoustics",n_fft_Acoustics_Error))
    n_fft_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(n_fft_entry_Acoustics, EntryList,n_fft_Acoustics_Error)).grid(row=4, column=2)
    n_fft_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    n_fft_moreinfo_Acoustics.grid(row=4, column=3, sticky=W)
    n_fft_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    n_fft_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    max_F0_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    max_F0_Acoustics_Error.grid(row=7, column=1)
    max_F0_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="max_F0").grid(row=6, column=0)
    max_F0_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    max_F0_entry_Acoustics.insert(0, "1830")
    max_F0_entry_Acoustics.grid(row=6, column=1)
    max_F0_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(max_F0_entry_Acoustics, "max_F0_entry_Acoustics",max_F0_Acoustics_Error))
    max_F0_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(max_F0_entry_Acoustics, EntryList,max_F0_Acoustics_Error)).grid(row=6, column=2)
    max_F0_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    max_F0_moreinfo_Acoustics.grid(row=6, column=3, sticky=W)
    max_F0_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    max_F0_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    min_frequency_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    min_frequency_Acoustics_Error.grid(row=9, column=1)
    min_frequency_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="min_frequency").grid(row=8, column=0)
    min_frequency_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    min_frequency_entry_Acoustics.insert(0, "380")
    min_frequency_entry_Acoustics.grid(row=8, column=1)
    min_frequency_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(min_frequency_entry_Acoustics, "min_frequency_entry_Acoustics",min_frequency_Acoustics_Error))
    min_frequency_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(min_frequency_entry_Acoustics, EntryList,min_frequency_Acoustics_Error)).grid(row=8, column=2)
    min_frequency_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    min_frequency_moreinfo_Acoustics.grid(row=8, column=3, sticky=W)
    min_frequency_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    min_frequency_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    freq_range_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    freq_range_Acoustics_Error.grid(row=11, column=1)
    freq_range_text_Acoustics=tk.Label(AcousticsSettingsFrame, text="freq_range").grid(row=10, column=0)
    freq_range_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    freq_range_entry_Acoustics.insert(0, "0.5")
    freq_range_entry_Acoustics.grid(row=10, column=1)
    freq_range_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(freq_range_entry_Acoustics, "freq_range_entry_Acoustics",freq_range_Acoustics_Error))
    freq_range_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(freq_range_entry_Acoustics, EntryList,freq_range_Acoustics_Error)).grid(row=10, column=2)
    freq_range_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    freq_range_moreinfo_Acoustics.grid(row=10, column=3, sticky=W)
    freq_range_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    freq_range_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    baseline_amp_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    baseline_amp_Acoustics_Error.grid(row=13, column=1)
    baseline_amp_text_Acoustics = tk.Label(AcousticsSettingsFrame, text="baseline_amp").grid(row=12, column=0)
    baseline_amp_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    baseline_amp_entry_Acoustics.insert(0, "70")
    baseline_amp_entry_Acoustics.grid(row=12, column=1)
    baseline_amp_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(baseline_amp_entry_Acoustics, "baseline_amp_entry_Acoustics",baseline_amp_Acoustics_Error))
    baseline_amp_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(baseline_amp_entry_Acoustics, EntryList,baseline_amp_Acoustics_Error)).grid(row=12, column=2)
    baseline_amp_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    baseline_amp_moreinfo_Acoustics.grid(row=12, column=3, sticky=W)
    baseline_amp_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    baseline_amp_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    fmax_yin_Acoustics_Error = tk.Label(AcousticsSettingsFrame, text="", font=ErrorFont)
    fmax_yin_Acoustics_Error.grid(row=15, column=1)
    fmax_yin_text_Acoustics=tk.Label(AcousticsSettingsFrame, text="fmax_yin").grid(row=14, column=0)
    fmax_yin_entry_Acoustics = tk.Entry(AcousticsSettingsFrame, justify="center")
    fmax_yin_entry_Acoustics.insert(0, "8000")
    fmax_yin_entry_Acoustics.grid(row=14, column=1)
    fmax_yin_entry_Acoustics.bind("<FocusOut>", lambda value:Validate_Settings(fmax_yin_entry_Acoustics, "fmax_yin_entry_Acoustics",fmax_yin_Acoustics_Error))
    fmax_yin_reset_Acoustics = tk.Button(AcousticsSettingsFrame, text="Reset", command=lambda:ResetAcousticSetting(fmax_yin_entry_Acoustics, EntryList,fmax_yin_Acoustics_Error)).grid(row=14, column=2)
    fmax_yin_moreinfo_Acoustics = tk.Button(AcousticsSettingsFrame, text='?', state="disabled", fg="black")
    fmax_yin_moreinfo_Acoustics.grid(row=14, column=3, sticky=W)
    fmax_yin_moreinfo_Acoustics.bind("<Enter>", MoreInfo)
    fmax_yin_moreinfo_Acoustics.bind("<Leave>", LessInfo)

    EntryList = [win_length_entry_Acoustics, hop_length_entry_Acoustics, n_fft_entry_Acoustics, max_F0_entry_Acoustics,
                 min_frequency_entry_Acoustics, freq_range_entry_Acoustics, baseline_amp_entry_Acoustics,
                 fmax_yin_entry_Acoustics]

    global AcousticErrorLabels
    AcousticErrorLabels = [win_length_Acoustics_Error,hop_length_Acoustics_Error,n_fft_Acoustics_Error,max_F0_Acoustics_Error,
                           min_frequency_Acoustics_Error,freq_range_Acoustics_Error,baseline_amp_Acoustics_Error,fmax_yin_Acoustics_Error]

    AcousticResetAllVariables = tk.Button(AcousticsSettingsFrame, text="Reset All", command=lambda:ResetAcousticSetting("all", EntryList,  "all"))
    AcousticResetAllVariables.grid(row=16, column=1)

    MultiAcousticsFileExplorer = tk.Button(AcousticsMainFrameMulti, text="Find Segmentation File",command=lambda: FileExplorer("Acoustics_Multi", "Input"))
    MultiAcousticsFileExplorer.grid(row=0, column=0)
    MultiAcousticsInputDir = tk.Label(AcousticsMainFrameMulti, text=Dir_Width*" ", bg="light grey")
    MultiAcousticsInputDir.grid(row=0, column=1, padx=Padding_Width)
    global MultiAcousticsBirdID
    MultiAcousticsBirdID = tk.Entry(AcousticsMainFrameMulti, font=("Arial", 15), justify="center", fg="grey",
                                   width=BirdID_Width)
    MultiAcousticsBirdID.insert(0, "Bird ID")
    MultiAcousticsBirdID.bind("<FocusIn>", focus_in)
    MultiAcousticsBirdID.bind("<FocusOut>", focus_out)
    MultiAcousticsBirdID.grid(row=2, column=1)

    MultiAcousticsOutputButton = tk.Button(AcousticsMainFrameMulti, text="Select Output Folder", command=lambda: FileExplorer("Acoustics_Multi", "Output"))
    MultiAcousticsOutputButton.grid(row=1, column=0)
    global MultiAcousticsOutputDisplay
    MultiAcousticsOutputDisplay = tk.Label(AcousticsMainFrameMulti, text=Dir_Width*" ", bg="light grey")
    MultiAcousticsOutputDisplay.grid(row=1, column=1, padx=Padding_Width)
    MultiAcousticsText = tk.Label(AcousticsMainFrameMulti, text="Please select which acoustics features you would like to analyze:")
    MultiAcousticsText.grid(row=3, column=0, columnspan=3)

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
    LoadLabelingSettings.grid(row=16, column=0)

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

    def MultiCheckAllBoxes(MultiCheckAll):
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

    MultiSettings_StartRow = 5
    MultiCheckAll_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text='Select All', variable=MultiCheckAll,
                                       command=lambda: CheckAllBoxes(MultiCheckAll,"Multi"))
    MultiCheckAll_CheckBox.grid(row=MultiSettings_StartRow, column=1, columnspan=1)
    MultiGoodness_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Goodness", anchor=tk.W, variable=MultiRunGoodness)
    MultiGoodness_CheckBox.grid(row=MultiSettings_StartRow+1, column=1, columnspan=1)
    MultiMean_frequency_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Mean Frequency", anchor=tk.W,
                                             variable=MultiRunMean_frequency)
    MultiMean_frequency_CheckBox.grid(row=MultiSettings_StartRow+2, column=1, columnspan=1)
    MultiEntropy_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Entropy", anchor=tk.W, variable=MultiRunEntropy)
    MultiEntropy_CheckBox.grid(row=MultiSettings_StartRow+3, column=1, columnspan=1)
    MultiAmplitude_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Amplitude", anchor=tk.W, variable=MultiRunAmplitude)
    MultiAmplitude_CheckBox.grid(row=MultiSettings_StartRow+4, column=1, columnspan=1)
    MultiAmplitude_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Amplitude Modulation", anchor=tk.W,
                                                   variable=MultiRunAmplitude_modulation)
    MultiAmplitude_modulation_CheckBox.grid(row=MultiSettings_StartRow+5, column=1, columnspan=1)
    MultiFrequency_modulation_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Frequency Modulation", anchor=tk.W,
                                                   variable=MultiRunFrequency_modulation)
    MultiFrequency_modulation_CheckBox.grid(row=MultiSettings_StartRow+6, column=1, columnspan=1)
    MultiPitch_CheckBox = tk.Checkbutton(AcousticsMainFrameMulti, text="Pitch", anchor=tk.W, variable=MultiRunPitch)
    MultiPitch_CheckBox.grid(row=MultiSettings_StartRow+7, column=1, columnspan=1)
    MultiAcousticsRunButton = tk.Button(AcousticsMainFrameMulti, text="Run", command=lambda: Acoustics_Syllables())
    MultiAcousticsRunButton.grid(row=MultiSettings_StartRow+8, column=1)

    #SegmentationHelpButton = tk.Button(SegmentationFrame, text="Help", command= lambda:HelpButton)
    #SegmentationHelpButton.grid(row=0, column=5, sticky="e")

    ### Syntax ###
    SyntaxFrame = tk.Frame(gui)
    notebook.add(SyntaxFrame, text="Syntax")

    SyntaxMainFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    SyntaxNotebook = ttk.Notebook(SyntaxFrame)
    SyntaxNotebook.grid(row=1)
    SyntaxNotebook.add(SyntaxMainFrame, text="Home")
    SyntaxSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    SyntaxNotebook.add(SyntaxSettingsFrame, text="Advanced Settings")
    # SyntaxInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    # SyntaxNotebook.add(SyntaxInfoFrame, text="Info")

    SyntaxFileButton = tk.Button(SyntaxMainFrame, text="Select Labeling File", command=lambda:FileExplorer("Syntax", "Input"))
    SyntaxFileButton.grid(row=1,column=0)
    global SyntaxFileDisplay
    SyntaxFileDisplay = tk.Label(SyntaxMainFrame, text=Dir_Width*" ", bg="light grey")
    SyntaxFileDisplay.grid(row=1, column=1, padx=Padding_Width, columnspan=3, sticky='w')
    SyntaxOutputButton = tk.Button(SyntaxMainFrame, text="Select Output Folder", command=lambda:FileExplorer("Syntax", "Output"))
    SyntaxOutputButton.grid(row=2, column=0)
    global SyntaxOutputDisplay
    SyntaxOutputDisplay = tk.Label(SyntaxMainFrame, text=Dir_Width*" ", bg="light grey")
    SyntaxOutputDisplay.grid(row=2, column=1, padx=Padding_Width, columnspan=3, sticky='w')
    SyntaxRunButton = tk.Button(SyntaxMainFrame, text="Run", command=lambda: Syntax())
    SyntaxRunButton.grid(row=5, column=0)
    SyntaxBirdID_Text = tk.Label(SyntaxMainFrame, text="Bird ID:").grid(row=3, column=0)
    global SyntaxBirdID
    SyntaxBirdID = tk.Entry(SyntaxMainFrame, fg="grey", font=("Arial",15), justify="center", width=BirdID_Width)
    SyntaxBirdID.insert(0, "Bird ID")
    SyntaxBirdID.bind("<FocusIn>", focus_in)
    SyntaxBirdID.bind("<FocusOut>", focus_out)
    SyntaxBirdID.grid(row=3, column=1, sticky='w', padx=Padding_Width)
    global DropCalls
    DropCalls = IntVar()
    DropCallsCheckbox = tk.Checkbutton(SyntaxMainFrame, text= "Drop Calls", variable=DropCalls)
    DropCallsCheckbox.grid(row=4)

    def ResetSyntaxSetting():
        min_gap_entry_Syntax.delete(0, END)
        min_gap_entry_Syntax.insert(0, "0.2")
        min_gap_entry_Syntax.config(bg='white')
        min_gap_entry_Syntax.update()
        min_gap_Syntax_Error.config(text="")

    min_gap_Syntax_Error = tk.Label(SyntaxSettingsFrame, text="", font=ErrorFont)
    min_gap_Syntax_Error.grid(row=1, column=1)
    min_gap_text_Syntax = tk.Label(SyntaxSettingsFrame, text="min_gap").grid(row=0, column=0)
    min_gap_entry_Syntax = tk.Entry(SyntaxSettingsFrame, justify="center")
    min_gap_entry_Syntax.insert(0, "0.2")
    min_gap_entry_Syntax.grid(row=0, column=1)
    min_gap_entry_Syntax.bind("<FocusOut>", lambda value:Validate_Settings(min_gap_entry_Syntax, "min_gap_entry_Syntax",min_gap_Syntax_Error))
    min_gap_reset_Syntax = tk.Button(SyntaxSettingsFrame, text="Reset",command=lambda: ResetSyntaxSetting()).grid(row=0, column=2)
    min_gap_moreinfo_Syntax = tk.Button(SyntaxSettingsFrame, text='?', state="disabled", fg="black")
    min_gap_moreinfo_Syntax.grid(row=0, column=3, sticky=W)
    min_gap_moreinfo_Syntax.bind("<Enter>", MoreInfo)
    min_gap_moreinfo_Syntax.bind("<Leave>", LessInfo)

    SyntaxDialogTitle = tk.Label(SyntaxSettingsFrame, text="", font=("Arial", 25, "bold"),justify="center")
    SyntaxDialogTitle.grid(row=0, column=4, rowspan=2)
    SyntaxDialog = tk.Label(SyntaxSettingsFrame, text="", justify="center")
    SyntaxDialog.grid(row=2, column=4, rowspan=2)


    Var_Heatmap = IntVar()
    SelectCountHeatmap = tk.Radiobutton(SyntaxMainFrame, text="Count Matrix", variable=Var_Heatmap, value=1)
    SelectCountHeatmap.grid(row=6, column=0)
    SelectProbHeatmap = tk.Radiobutton(SyntaxMainFrame, text="Probability Matrix", variable=Var_Heatmap, value=2)
    SelectProbHeatmap.grid(row=7, column=0)

    def GenerateMatrixHeatmap():
        global syntax_data
        global Bird_ID
        if Var_Heatmap.get() == 1:
            try:
                # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
                fig = Figure()
                sns.heatmap(syntax_data.trans_mat, annot=True, fmt='0.0f')
            except:
                Syntax()
                # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
                fig = Figure()
                sns.heatmap(syntax_data.trans_mat, annot=True, fmt='0.0f')
            plt.title("Count Transition Matrix")
            plt.xticks(rotation=0)
            plt.savefig("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Counts.png")
            Heatmap = PhotoImage(file="C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Counts.png")
        if Var_Heatmap.get() == 2:
            try:
                # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
                sns.heatmap(syntax_data.trans_mat_prob, annot=True, fmt='0.2f')
            except:
                Syntax()
                # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
                sns.heatmap(syntax_data.trans_mat_prob, annot=True, fmt='0.2f')
            plt.title("Probability Transition Matrix")
            plt.xticks(rotation=0)
            plt.savefig("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Prob.png")
            Heatmap = PhotoImage(file="C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Prob.png")
        MatrixHeatmap_Display = tk.Toplevel(gui)
        MatrixHeatmap_Display_Label = tk.Label(MatrixHeatmap_Display, image= Heatmap)
        MatrixHeatmap_Display_Label.grid(row=1, column=0, columnspan=3)
        #tbar = fig.canvas.toolbar
        #tbar.add_button(testbutton)
        #fig.canvas.draw()
        #plt.show()
        Type = ""
        if Var_Heatmap.get() == 1:
            Type = "Count"
        if Var_Heatmap.get() == 2:
            Type = "Prob"
        MatrixHeatmap_SaveButton = tk.Button(MatrixHeatmap_Display, text="Save", command=lambda:plt.savefig("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/"+str(Bird_ID)+"_"+Type+"_MatrixHeatmap.png"))
        MatrixHeatmap_SaveButton.grid(row=0, column=1)
        try:
            os.remove("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Prob.png")
        except: pass
        try:
            os.remove("C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/A321/MatrixHeatmap_Counts.png")
        except: pass
        plt.clf()
        MatrixHeatmap_Display.mainloop()

    MatrixHeatmap_Button = tk.Button(SyntaxMainFrame, text="Generate Matrix Heatmap", command=lambda:GenerateMatrixHeatmap())
    MatrixHeatmap_Button.grid(row=6, column=1, rowspan=2)

    def GenerateRasterPlot():
        global syntax_data
        global Bird_ID
        global SyntaxDirectory
        global SyntaxBirdID
        try: SyntaxAlignment_WarningMessage.destroy()
        except: pass
        SyntaxOutputFolder = SyntaxOutputDisplay.cget("text") + "/Syntax_Plots/"
        try:
            os.makedirs(SyntaxOutputFolder)
        except: pass
        if SyntaxAlignmentVar.get() == "Select Label":
            SyntaxAlignment_WarningMessage = tk.Label(SyntaxMainFrame, text="Warning: Must Select Label to Proceed", font=("Arial", 7))
            SyntaxAlignment_WarningMessage.grid(row=8, column=1)
        RasterOutputDirectory = ""
        for i in SyntaxDirectory.split("/")[:-1]:
            RasterOutputDirectory = RasterOutputDirectory+i+"/"
        else:
            if ChangeFigSize.get() == 1:
                X = int(X_Size.get())
                Y = int(Y_Size.get())
            else:
                X = 6
                Y = 6

            if SyntaxAlignmentVar.get() != "Auto":
                try:
                    syntax_raster_df = syntax_data.make_syntax_raster(alignment_syllable=int(SyntaxAlignmentVar.get()))
                except:
                    Syntax()
                    syntax_raster_df = syntax_data.make_syntax_raster(alignment_syllable=int(SyntaxAlignmentVar.get()))
                try:
                    FigTitle = str(Bird_ID)+" Syntax Raster_Syl"+str(SyntaxAlignmentVar.get())
                except:
                    Bird_ID = SyntaxBirdID.get()
                    FigTitle = str(Bird_ID)+" Syntax Raster_Syl"+str(SyntaxAlignmentVar.get())

                ### Note for Therese: I added "return plt" to line 1083 of avn.syntax and line 139 to avn.plotting

                fig = avn.plotting.plot_syntax_raster(syntax_data, syntax_raster_df, title=FigTitle, figsize=(X, Y))
            if SyntaxAlignmentVar.get() == "Auto":
                try:
                    syntax_raster_df = syntax_data.make_syntax_raster()
                except:
                    Syntax()
                    syntax_raster_df = syntax_data.make_syntax_raster()
                try:
                    FigTitle = str(Bird_ID)+" Syntax Raster"
                except:
                    Bird_ID = SyntaxBirdID.get()
                    FigTitle = str(Bird_ID)+" Syntax Raster"
                print(syntax_data)
                print("-------------")
                print(syntax_raster_df)
                fig = avn.plotting.plot_syntax_raster(syntax_data, syntax_raster_df, title=FigTitle, figsize=(X, Y))

            #### ISSUE: Raster plot function currently creates an empty raster plot (and saved figure is also empty)



            fig.savefig(SyntaxOutputFolder+"raster_fig.png")
            DisplayRaster = tk.Toplevel(gui)
            SaveRaster = tk.Button(DisplayRaster, text="Save", command=lambda:avn.plotting.plot_syntax_raster(syntax_data, syntax_raster_df, title=FigTitle).savefig(RasterOutputDirectory+FigTitle+".png"))
            SaveRaster.grid(row=0, column=1)
            label_canvas = FigureCanvasTkAgg(fig, master=DisplayRaster)  # A tk.DrawingArea.
            label_canvas.draw()
            label_canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

    global AlignmentChoices
    global SyntaxAlignmentVar
    global SyntaxAligment
    SyntaxAlignmentVar = StringVar()
    SyntaxAlignmentVar.set("Select Label")
    AlignmentChoices = ["Select Label"]
    SyntaxAlignment = tk.OptionMenu(SyntaxMainFrame, SyntaxAlignmentVar, *AlignmentChoices)
    SyntaxAlignment.grid(row=8, column=0)

    RasterButton = tk.Button(SyntaxMainFrame, text="Generate Raster Plot", command=lambda:GenerateRasterPlot())
    RasterButton.grid(row=8, column=1)

    def Save_Syllable_Stats():
        global SyntaxDirectory
        global Bird_ID
        global syntax_data
        OutputDir = ""
        for i in SyntaxDirectory.split("/")[:-1]:
            OutputDir = OutputDir+i+"/"
        try:
            entropy_rate = syntax_data.get_entropy_rate()
        except:
            Syntax()
            entropy_rate = syntax_data.get_entropy_rate()
        entropy_rate_norm = entropy_rate / np.log2(len(syntax_data.unique_labels) + 1)

        single_rep_counts, single_rep_stats = syntax_data.get_single_repetition_stats()
        intro_notes_df = syntax_data.get_intro_notes_df()
        prop_sylls_in_short_bouts = syntax_data.get_prop_sylls_in_short_bouts(max_short_bout_len=2)
        per_syll_stats = syntax.Utils.merge_per_syll_stats(single_rep_stats, prop_sylls_in_short_bouts, intro_notes_df)
        per_syll_stats = per_syll_stats.set_index('syllable')

        single_rep_counts = single_rep_counts.set_index('syllable')
        single_rep_counts=single_rep_counts.drop(['Bird_ID'], axis=1)

        prob_repetitions = syntax_data.get_prob_repetitions()
        drop = prob_repetitions['syllable'] == 'silent_gap'
        prob_repetitions = prob_repetitions[~drop]
        prob_repetitions = prob_repetitions.set_index('syllable')
        prob_repetitions = prob_repetitions.drop(['Bird_ID'], axis=1)

        per_syll_stats = pd.concat([per_syll_stats, prob_repetitions, single_rep_counts], axis=1)

        per_syll_stats.loc[per_syll_stats.index[0], 'entropy_rate'] = entropy_rate
        per_syll_stats.loc[per_syll_stats.index[0], 'entropy_rate_norm'] = entropy_rate_norm

        print(per_syll_stats)
        per_syll_stats.to_csv(OutputDir+Bird_ID+"_SyllableStats.csv")

    Syllable_Stats = tk.Button(SyntaxMainFrame, text="Save Syllable Stats", command=lambda:Save_Syllable_Stats())
    Syllable_Stats.grid(row=9, column=0)

    X_Size = StringVar()
    X_Size.set("10")
    FixSize_X_Text = tk.Label(SyntaxMainFrame, text="X:").grid(row=4, column=2, sticky="e", padx=10)
    FigSize_X = tk.Spinbox(SyntaxMainFrame, from_=1, to=20, state=DISABLED, textvariable=X_Size, width=15)
    FigSize_X.grid(row=4, column=3, padx=10)

    Y_Size = StringVar()
    Y_Size.set("10")
    FixSize_Y_Text = tk.Label(SyntaxMainFrame, text="Y:").grid(row=5, column=2, sticky="e", padx=10)
    FigSize_Y = tk.Spinbox(SyntaxMainFrame, from_=1, to=20, state=DISABLED, textvariable=Y_Size, width=15)
    FigSize_Y.grid(row=5, column=3, padx=10)

    def ChangeFigureSize():
        if ChangeFigSize.get() == 1:
            FigSize_X.config(state=NORMAL)
            FigSize_Y.config(state=NORMAL)
        if ChangeFigSize.get() == 0:
            FigSize_X.config(state=DISABLED)
            FigSize_Y.config(state=DISABLED)

    ChangeFigSize = IntVar()
    CustomFigSize = tk.Checkbutton(SyntaxMainFrame, text="Custom Figure Size", variable=ChangeFigSize, command=lambda:ChangeFigureSize())
    CustomFigSize.grid(row=6, column=2, columnspan=2, padx=10)

    ### Plain Spectrogram Generation - Whole File ###
    PlainSpectrograms = tk.Frame(gui, width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(PlainSpectrograms, text="Plain Spectrograms")

    Plain_Folder = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook = ttk.Notebook(PlainSpectrograms)
    PlainNotebook.grid(row=1)
    PlainNotebook.add(Plain_Folder, text="Whole Folder")

    global PlainDirectoryLabel
    PlainDirectoryLabel = tk.Label(Plain_Folder, text=Dir_Width*" ", bg="light grey")
    PlainDirectoryLabel.grid(row=1, column=1, padx=Padding_Width)

    PlainFileExplorer = tk.Button(Plain_Folder, text="Select Folder", command=lambda: FileExplorer("Plain_Folder", "Input"))
    PlainFileExplorer.grid(row=1, column=0)
    global PlainOutputFolder_Label
    PlainOutputFolder_Label = tk.Label(Plain_Folder, text=Dir_Width*" ", bg="light grey")
    PlainOutputFolder_Label.grid(row=2, column=1, padx=Padding_Width)
    PlainOutputFolder_Button = tk.Button(Plain_Folder, text="Select Output Folder",
                                         command=lambda: FileExplorer("Plain_Folder", "Output"))
    PlainOutputFolder_Button.grid(row=2, column=0)
    PlainSpectroRun = tk.Button(Plain_Folder, text="Create Blank Spectrograms", command=lambda:PrintPlainSpectrograms())
    PlainSpectroRun.grid(row=3, column=0, columnspan=2)
    
    ### Plain Spectrogram Generation - Selected Files ###
    PlainSpectroAlt = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    PlainNotebook.add(PlainSpectroAlt, text="Individual Files")
    PlainFileExplorerAlt = tk.Button(PlainSpectroAlt, text="Select Files", command=lambda: FileExplorer("Plain_Files", "Input"))
    PlainFileExplorerAlt.grid(row=1, column=0)
    PlainDirectoryLabelAlt = tk.Label(PlainSpectroAlt, text=Dir_Width*" ", bg="light grey")
    PlainDirectoryLabelAlt.grid(row=1, column=1, padx=Padding_Width)
    PlainOutputAlt_Button = tk.Button(PlainSpectroAlt, text="Select Output Folder", command=lambda:FileExplorer("Plain_Files", "Output"))
    PlainOutputAlt_Button.grid(row=2, column = 0)

    global PlainOutputAlt_Label
    PlainOutputAlt_Label = tk.Label(PlainSpectroAlt, text = Dir_Width*" ", bg="light grey")
    PlainOutputAlt_Label.grid(row=2, column=1, padx=Padding_Width)
    PlainSpectroRunAlt = tk.Button(PlainSpectroAlt, text="Create Blank Spectrograms",
                                command=lambda: PrintPlainSpectrogramsAlt())
    PlainSpectroRunAlt.grid(row=3, column=0, columnspan=2)
    # PlainSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    # PlainNotebook.add(PlainSettingsFrame, text="Advanced Settings")
    # PlainInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    # PlainNotebook.add(PlainInfoFrame, text="Info")
    
    ### Timing Module ###
    TimingTab = tk.Frame(gui, width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(TimingTab, text="Timing")
    TimingNotebook = ttk.Notebook(TimingTab)
    TimingNotebook.grid(row=1)
    TimingMainFrame = tk.Frame(TimingTab, width=MasterFrameWidth, height=MasterFrameHeight)
    TimingNotebook.add(TimingMainFrame, text="Home")
    # TimingSettings = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    # TimingNotebook.add(TimingSettings, text="Advanced Settings")

    TimingInput_Button = tk.Button(TimingMainFrame, text="Find Segmentations", command=lambda:FileExplorer("Timing", "Input"))
    TimingInput_Button.grid(row=0, column=0)
    global TimingInput_Text
    TimingInput_Text = tk.Label(TimingMainFrame, text=Dir_Width*" ", bg="light grey")
    TimingInput_Text.grid(row=0, column=1, columnspan=2, padx=Padding_Width)
    TimingOutput_Button = tk.Button(TimingMainFrame, text="Select Output Folder", command=lambda:FileExplorer("Timing", "Output"))
    TimingOutput_Button.grid(row=1, column=0)
    global TimingOutput_Text
    TimingOutput_Text = tk.Label(TimingMainFrame, text=Dir_Width * " ", bg="light grey")
    TimingOutput_Text.grid(row=1, column=1, columnspan=2, padx=Padding_Width)
    Timing_BirdID = tk.Entry(TimingMainFrame, font=("Arial", 15), justify="center", fg="grey",
                                   width=BirdID_Width)
    Timing_BirdID.insert(0, "Bird ID")
    Timing_BirdID.bind("<FocusIn>", focus_in)
    Timing_BirdID.bind("<FocusOut>", focus_out)
    Timing_BirdID.grid(row=2, column=0, columnspan=3, pady=5)

    RunTiming = tk.Button(TimingMainFrame, text="Run Timing", command=lambda:Timing())
    RunTiming.grid(row=3, column=0, columnspan=3)

    # Run Everything Tab #
    RunAllTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
    notebook.add(RunAllTab, text="Run All")
    RunAll_Title = tk.Label(RunAllTab, text="This will calculate the complete AVN feature set \n"
                                            "for all labeled songs in the selected label table. \n \n"
                                            "Settings and parameters for these features can be adjusted \n"
                                            "within each feature's dedicated tab", font=("Arial",15)).grid(row=0, columnspan=2)
    RunAllPadding = tk.Label(RunAllTab, text="", font=("Arial",20)).grid(row=1)

    RunAll_ImportantFrame = tk.Frame(RunAllTab, width=400, height=200, highlightbackground="black", highlightthickness=1)
    RunAll_ImportantFrame.grid(row=2, column=0, columnspan=2)
    RunAll_Input = tk.Button(RunAll_ImportantFrame, text="Input labeling file:", command=lambda:FileExplorer("RunAll","Input"))
    RunAll_Input.grid(row=2, column=0)
    RunAll_InputFileDisplay = tk.Label(RunAll_ImportantFrame, text=Dir_Width*" ", bg="light grey")
    RunAll_InputFileDisplay.grid(row=2, column=1)

    RunAll_BirdID_Label = tk.Label(RunAll_ImportantFrame, text="Bird ID:").grid(row=3, column=0)
    RunAll_BirdID_Entry = tk.Entry(RunAll_ImportantFrame, font=("Arial", 15), justify="center", fg="grey", bg="white",
                                   width=BirdID_Width)
    RunAll_BirdID_Entry.grid(row=3, column=1)
    RunAll_BirdID_Entry.insert(0, "Bird ID")
    RunAll_BirdID_Entry.bind("<FocusIn>", focus_in)
    RunAll_BirdID_Entry.bind("<FocusOut>", focus_out)

    RunAll_Output = tk.Button(RunAll_ImportantFrame, text="Output Destination:",command=lambda:FileExplorer("RunAll","Output"))
    RunAll_Output.grid(row=4, column=0)
    RunAll_OutputFileDisplay = tk.Label(RunAll_ImportantFrame, text=Dir_Width*" ", bg="light grey")
    RunAll_OutputFileDisplay.grid(row=4, column=1)

    RunAllPadding2 = tk.Label(RunAllTab, text="", font=("Arial",20)).grid(row=5)

    RunAll_StartRow = 6
    #
    # global RunAll_CheckAll
    # global RunAll_RunGoodness
    # global RunAll_RunMean_frequency
    # global RunAll_RunEntropy
    # global RunAll_RunAmplitude
    # global RunAll_RunAmplitude_modulation
    # global RunAll_RunFrequency_modulation
    # global RunAll_RunPitch
    #
    # RunAll_CheckAll = IntVar()
    # RunAll_RunGoodness = IntVar()
    # RunAll_RunMean_frequency = IntVar()
    # RunAll_RunEntropy = IntVar()
    # RunAll_RunAmplitude = IntVar()
    # RunAll_RunAmplitude_modulation = IntVar()
    # RunAll_RunFrequency_modulation = IntVar()
    # RunAll_RunPitch = IntVar()
    #
    # RunAll_AcousticsLabel = tk.Label(RunAllTab, text="Select Acoustic Features to analyze:").grid(row=RunAll_StartRow-1)
    # RunAll_CheckAllAcoustics = tk.Checkbutton(RunAllTab, text="Select All", variable=RunAll_CheckAll,
    #                                    command=lambda: CheckAllBoxes(RunAll_CheckAll,"RunAll"))
    # RunAll_CheckAllAcoustics.grid(row=RunAll_StartRow, column=0)
    # RunAll_Goodness_CheckBox = tk.Checkbutton(RunAllTab, text="Goodness", anchor=tk.W, variable=RunAll_RunGoodness)
    # RunAll_Goodness_CheckBox.grid(row=RunAll_StartRow + 1, column=0)
    # RunAll_Mean_frequency_CheckBox = tk.Checkbutton(RunAllTab, text="Mean Frequency", anchor=tk.W,variable=RunAll_RunMean_frequency)
    # RunAll_Mean_frequency_CheckBox.grid(row=RunAll_StartRow + 2, column=0)
    # RunAll_Entropy_CheckBox = tk.Checkbutton(RunAllTab, text="Entropy", anchor=tk.W, variable=RunAll_RunEntropy)
    # RunAll_Entropy_CheckBox.grid(row=RunAll_StartRow + 3, column=0)
    # RunAll_Amplitude_CheckBox = tk.Checkbutton(RunAllTab, text="Amplitude", anchor=tk.W, variable=RunAll_RunAmplitude)
    # RunAll_Amplitude_CheckBox.grid(row=RunAll_StartRow + 4, column=0)
    # RunAll_Amplitude_modulation_CheckBox = tk.Checkbutton(RunAllTab, text="Amplitude Modulation", anchor=tk.W,variable=RunAll_RunAmplitude_modulation)
    # RunAll_Amplitude_modulation_CheckBox.grid(row=RunAll_StartRow + 5, column=0)
    # RunAll_Frequency_modulation_CheckBox = tk.Checkbutton(RunAllTab, text="Frequency Modulation", anchor=tk.W,variable=RunAll_RunFrequency_modulation)
    # RunAll_Frequency_modulation_CheckBox.grid(row=RunAll_StartRow + 6, column=0)
    # RunAll_Pitch_CheckBox = tk.Checkbutton(RunAllTab, text="Pitch", anchor=tk.W, variable=RunAll_RunPitch)
    # RunAll_Pitch_CheckBox.grid(row=RunAll_StartRow + 7, column=0)
    #
    # global RunAll_DropCalls
    # RunAll_DropCalls = IntVar()
    # RunAll_AcousticsTitle = tk.Label(RunAllTab, text="Syntax Settings:").grid(row=RunAll_StartRow-1, column=1)
    # RunAll_DropCalls = tk.Checkbutton(RunAllTab, text="Drop Calls", variable=RunAll_DropCalls)
    # RunAll_DropCalls.grid(row=RunAll_StartRow, column=1)
    # global RunAll_CountMatrix
    # RunAll_CountMatrix = IntVar()
    # RunAll_GenerateCountMatrix = tk.Checkbutton(RunAllTab, text="Generate Matrix Heatmap", variable=RunAll_CountMatrix)
    # RunAll_GenerateCountMatrix.grid(row=RunAll_StartRow+1, column=1)
    # global RunAll_ProbabilityMatrix
    # RunAll_ProbabilityMatrix = IntVar()
    # RunAll_GenerateProbabilityMatrix = tk.Checkbutton(RunAllTab, text="Generate Probability Heatmap", variable=RunAll_ProbabilityMatrix)
    # RunAll_GenerateProbabilityMatrix.grid(row=RunAll_StartRow + 2, column=1)
    # global RunAll_Raster
    # RunAll_Raster = IntVar()
    # RunAll_GenerateRaster = tk.Checkbutton(RunAllTab, text="Generate Raster Plot",
    #                                                   variable=RunAll_Raster)
    # RunAll_GenerateRaster.grid(row=RunAll_StartRow + 3, column=1)

    def RunAllModules():
        print("test")
    RunAll_RunButton = tk.Button(RunAllTab, text="Run", command=lambda:RunAllModules())
    RunAll_RunButton.grid(row=RunAll_StartRow+8, column=0, columnspan=2)

    ttk.Style().theme_use("clam")
    gui.mainloop()

except Exception:
    print(Exception)