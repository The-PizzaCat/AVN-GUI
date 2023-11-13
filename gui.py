'''
Things to change/update from Therese:
#DONE# 1. Generate spectrograms without needing to segment -- Do this last; super difficult
#DONE# 2. Display thresholds on spectrograms -- use axis data from fig object to add horizontal line at min/max thresholds
#DONE# 3. Change dimensions of spectrogram images by modifying hidden variable from avn segmentation
#DONE# 4. Add arrow buttons in spectrogram window that changes index of file selected for spectrogram instead of using tabs
	-- Update tab upon clicking arrow button
#DONE# 5. Have segmentation tables have unique names so that they don't get overwritten
	Name is based on Bird ID + Min Threshold + Max Threshold



For Labeling Env:
1. Select Folder w/ song
2. Select Output segmentations
3. Run!

***Gui eventually should have a tab for segmentation and one for labeling***
***Add brief instructions for each tab (seg vs labeling)***
'''


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
    import librosa
    import shutil
    import os

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
        except:
            pass

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

    gui = tk.Tk()
    gui.title("AVN Segmentation")
    greeting = tk.Label(text="Welcome to the AVN gui!")
    greeting.grid(row=0, columnspan=2)

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
    SegmentButton.grid(row=6, columnspan=2)

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

    SpectrogramButton = tk.Button(text="Create Spectrogram", command = lambda : SpectrogramDisplay("Start"))
    SpectrogramButton.grid(row=5, columnspan=2)

    def labelingPlaceholder():
        ################# LABELING #######################
        '''
        global Bird_ID
        global segmentations_path
        segmentations_path = str(Bird_ID)+"_seg_table.csv" #e.g. "C:/where_my_segmentation_table_is/Bird_ID_segmentations.csv"
        global song_folder
        song_folder_path = song_folder #e.g. "C:/where_my_wav_files_are/Bird_ID/90/"
        output_file = song_folder+"/"+str(Bird_ID)+"_labels.csv" #e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"
        '''
        '''
        import pymde
        import numpy as np
        import pandas as pd
        import librosa
        import matplotlib.pyplot as plt
        import avn.dataloading
        import avn.plotting
        import hdbscan
        import math
        import sklearn
        import seaborn as sns
        '''
        '''
        def make_spec(syll_wav, hop_length = 128, win_length = 512, n_fft = 512, amin = 1e-5, ref_db = 20, min_level_db = -28):
            spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
            spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)
    
            # normalize
            S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)
    
            return S_norm
    
        labelingWindow = tk.Toplevel()
        LabelingButton = tk.Button(labelingWindow, command = lambda : make_spec(syll_wav='C:/Users/ethan/Desktop/Roberts_Lab_Files/AVNGUI/B715/B715_44831.28518883_9_27_7_55_18.wav'))
        LabelingButton.pack()
    
        amin = 1e-5
        ref_db = 20
        min_level_db = -28
        win_length = 512  # the AVGN default is based on ms duration and sample rate, and it 441.
        hop_length = 128  # the AVGN default is based on ms duration and sample rate, and it is 88.
        n_fft = 512
        K = 10
        min_cluster_prop = 0.04
        embedding_dim = 2
        '''
        '''
        # load segmentations
        predictions_reformat = pd.read_csv(segmentations_path)
        predictions_reformat = predictions_reformat.rename(columns={"onset_s": 'onsets',
                                                                    "offset_s": 'offsets',
                                                                    "audio_path": 'files'})
        predictions_reformat = predictions_reformat[predictions_reformat.label == 's']
    
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
    
        for song_file in segmentations.files.unique():
            file_path = song_folder_path + song_file
            song = avn.dataloading.SongFile(file_path)
            song.bandpass_filter(500, 15000)
    
            syllable_df = segmentations[segmentations['files'] == song_file]
    
            # this section is based on avgn.signalprocessing.create_spectrogram_dataset.get_row_audio()
            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                    for st, et in zip(syllable_df.onsets.values, syllable_df.offsets.values)]
            syllable_dfs = pd.concat([syllable_dfs, syllable_df])
    
        # Normalize the audio
        syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
    
        # compute spectrogram for each syllable
        syllables_spec = []
    
        for syllable in syllable_dfs.audio.values:
    
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
    
        hdbscan_df = syllable_dfs
        hdbscan_df["labels"] = clusterer.labels_
    
        hdbscan_df["X"] = embedding[:, 0]
        hdbscan_df["Y"] = embedding[:, 1]
        hdbscan_df["labels"] = hdbscan_df['labels'].astype("category")
    
        hdbscan_df.to_csv(output_file)
    
        sns.scatterplot(data=hdbscan_df, x="X", y="Y", hue="labels", alpha=0.25, s=5)
        plt.title("My Bird's Syllables");
    
        for i in range(3):
            avn.plotting.plot_spectrogram_with_labels(hdbscan_df, "./", Bird_ID, song_file_index=i, figsize=(20, 5),
                                                      fontsize=14)
        '''




    #def on_closing():
    #    files = glob.glob(song_folder + "/TempSpectrogramFiles")
    #    for f in files:
    #        os.remove(f)
    #    gui.destroy()
    #gui.protocol("WM_DELETE_WINDOW", on_closing)

    gui.mainloop()


except Exception:
    print(Exception)