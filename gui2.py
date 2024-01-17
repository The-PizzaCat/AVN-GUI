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
    import csv
    import numpy as np
    import glob
    import re
    import pandas as pd

    pd.options.mode.chained_assignment = None
    import librosa
    #import shutil
    import os
    import time
    #import customtkinter
    #from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    #import pymde
    #import matplotlib
    #import matplotlib.pyplot as plt
    #import hdbscan
    #import math
    #import sklearn
    #import seaborn as sns
    #from matplotlib.figure import Figure

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

    def SegmentButtonClick(): ### This function has been modified to use Whisper via cpu
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
        sys.path.insert(0, 'C:/Users/ethan/Desktop/WhisperSeg-master')

        from model import WhisperSegmenterFast # This line sometimes doesn't work. if so, do pip install for missing module
        segmenter = WhisperSegmenterFast("nccratliri/whisperseg-large-ms-ct2", device="cpu")

        # get list of files to segment
        song_folder_path = song_folder

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
            curr_prediction_df['directory'] = song[:-(len(song_name))].replace("\\","/")
            # add current file's segments to full_seg_table
            full_seg_table = pd.concat([full_seg_table, curr_prediction_df])
            Progress_Var.set(i)
            ProgressBar.update()
            time.sleep(0.1)

        SegmentingProgressLabel.config(text="Segmentation Complete!")
        ProgressBar.destroy()
        time.sleep(1)
        # save full_seg_table
        for i in full_seg_table.directory.unique():
            print(i)
            tempTable = full_seg_table[full_seg_table.directory == i]
            try:
                tempTable.to_csv(str(i) + str(Bird_ID) + i.split("/")[-1]+"_wseg.csv")
                SegmentingProgressLabel.config(text="Successfully Saved Segmentation Data!")
                #full_seg_table = None
                tempTable = None
                prediction = None
            except:
                SegmentingProgressLabel.config(text="Failed to Save Segmentation Data")

        SegmentingProgressLabel.update()

    '''
    def MoreInfo(Event):
        # print(Event.widget)
        if str(Event.widget) == ".!frame15.!button2":  # win_length
            AcousticSettingsDialog.config(text="Length of window over which to calculate each \n feature in samples")
            AcousticsSettingsDialogTitle.config(text='win_length')
        if str(Event.widget) == ".!frame15.!button4":  # hop_length
            AcousticSettingsDialog.config(text="Number of samples to advance between windows")
            AcousticsSettingsDialogTitle.config(text='hop_length')
        if str(Event.widget) == ".!frame15.!button6":  # n_fft
            AcousticSettingsDialog.config(text="Length of the transformed axis of the output. \n If n is smaller than " \
                                               "the length of \n the win_length, the input is cropped")
            AcousticsSettingsDialogTitle.config(text='n_fft')
        if str(Event.widget) == ".!frame15.!button8":  # max_F0
            AcousticSettingsDialog.config(text="Maximum allowable fundamental frequency \n of signal in Hz")
            AcousticsSettingsDialogTitle.config(text='max_F0')
        if str(Event.widget) == ".!frame15.!button10":  # min_frequency
            AcousticSettingsDialog.config(text="Lower frequency cutoff in Hz. Only power at \n frequencies above " \
                                               "this will contribute to \n feature calculation")
            AcousticsSettingsDialogTitle.config(text='min_frequency')
        if str(Event.widget) == ".!frame15.!button12":  # freq_range
            AcousticSettingsDialog.config(text="Proportion of power spectrum \n frequency bins to consider")
            AcousticsSettingsDialogTitle.config(text='freq_range')
        if str(Event.widget) == ".!frame15.!button14":  # baseline_amp
            AcousticSettingsDialog.config(text="Baseline amplitude used to \n calculate amplitude in dB")
            AcousticsSettingsDialogTitle.config(text='baseline_amp')
        if str(Event.widget) == ".!frame15.!button16":  # fmax_yin
            AcousticSettingsDialog.config(text="Maximum frequency in Hz used to \n estimate fundamental frequency " \
                                               "with \n the YIN algorithm")
            AcousticsSettingsDialogTitle.config(text='fmax_yin')

        AcousticSettingsDialog.update()
        ##############################################################################3
        if str(Event.widget) == ".!frame8.!button2":  # bandpass_lower_cutoff
            LabelingSpectrogramDialog.config(text="Lower cutoff frequency in Hz for a hamming window \n" \
                                                  "bandpass filter applied to the audio data before generating\n" \
                                                  "spectrograms. Frequencies below this value will be filtered out")
        if str(Event.widget) == ".!frame8.!button4":  # bandpass_upper_cutoff
            LabelingSpectrogramDialog.config(text="Upper cutoff frequency in Hz for a hamming window bandpass\n" \
                                                  " filter applied to the audio data before generating spectrograms. \n" \
                                                  "Frequencies above this value will be filtered out")
        if str(Event.widget) == ".!frame8.!button6":  # a_min
            LabelingSpectrogramDialog.config(text="Minimum amplitude threshold in the spectrogram. Values \n" \
                                                  "lower than a_min will be set to a_min before conversion to decibels")
        if str(Event.widget) == ".!frame8.!button8":  # ref_db
            LabelingSpectrogramDialog.config(text="When making the spectrogram and converting it from amplitude \n" \
                                                  "to db, the amplitude is scaled relative to this reference: \n" \
                                                  "20 * log10(S/ref_db) where S represents the spectrogram with amplitude values")
        if str(Event.widget) == ".!frame8.!button10":  # min_level_db
            LabelingSpectrogramDialog.config(
                text="When making the spectrogram, once the amplitude has been converted \n" \
                     "to decibels, the spectrogram is normalized according to this value: \n" \
                     "(S - min_level_db)/-min_level_db where S represents the spectrogram \n" \
                     "in db. Any values of the resulting operation which are <0 are set to \n" \
                     "0 and any values that are >1 are set to 1")
        if str(Event.widget) == ".!frame8.!button12":  # n_fft
            LabelingSpectrogramDialog.config(text="When making the spectrogram, this is the length of the windowed \n" \
                                                  "signal after padding with zeros. The number of rows spectrogram is\n" \
                                                  " \"(1+n_fft/2)\". The default value,\"n_fft=512\" samples, \n" \
                                                  "corresponds to a physical duration of 93 milliseconds at a sample \n" \
                                                  "rate of 22050 Hz, i.e. the default sample rate in librosa. This value \n" \
                                                  "is well adapted for music signals. However, in speech processing, the \n" \
                                                  "recommended value is 512, corresponding to 23 milliseconds at a sample\n" \
                                                  " rate of 22050 Hz. In any case, we recommend setting \"n_fft\" to a \n" \
                                                  "power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm")
        if str(Event.widget) == ".!frame8.!button14":  # win_length
            LabelingSpectrogramDialog.config(
                text="When making the spectrogram, each frame of audio is windowed by a window \n" \
                     "of length \"win_length\" and then padded with zeros to match \"n_fft\".\n" \
                     " Padding is added on both the left- and the right-side of the window so\n" \
                     " that the window is centered within the frame. Smaller values improve \n" \
                     "the temporal resolution of the STFT (i.e. the ability to discriminate \n" \
                     "impulses that are closely spaced in time) at the expense of frequency \n" \
                     "resolution (i.e. the ability to discriminate pure tones that are closely\n" \
                     " spaced in frequency). This effect is known as the time-frequency \n" \
                     "localization trade-off and needs to be adjusted according to the \n" \
                     "properties of the input signal")
        if str(Event.widget) == ".!frame8.!button16":  # hop_length
            LabelingSpectrogramDialog.config(
                text="The number of audio samples between adjacent windows when creating \n" \
                     "the spectrogram. Smaller values increase the number of columns in \n" \
                     "the spectrogram without affecting the frequency resolution")
        if str(Event.widget) == ".!frame8.!button18":  # max_spec_size
            LabelingSpectrogramDialog.config(text="Maximum frequency in Hz used to \n estimate fundamental frequency " \
                                                  "with \n the YIN algorithm")
        if str(Event.widget) == ".!frame9.!button2":  # n_neighbors
            LabelingUMAPDialog.config(
                text="The size of local neighborhood (in terms of number of neighboring sample points)\n" \
                     " used for manifold approximation. Larger values result in more global views \n" \
                     "of the manifold, while smaller values result in more local data being \n" \
                     "preserved. In general values should be in the range 2 to 100")
        if str(Event.widget) == ".!frame9.!button4":  # n_components
            LabelingUMAPDialog.config(text="The dimension of the space to embed into. This defaults to 2 to provide \n" \
                                           "easy visualization, but can reasonably be set to any integer value \n" \
                                           "in the range 2 to 100")
        if str(Event.widget) == ".!frame9.!button6":  # min_dist
            LabelingUMAPDialog.config(text="The effective minimum distance between embedded points. \n" \
                                           "Smaller values will result in a more clustered/clumped \n" \
                                           "embedding where nearby points on the manifold are drawn \n" \
                                           "closer together, while larger values will result on a more \n" \
                                           "even dispersal of points. The value should be set relative to \n" \
                                           "the \"spread\" value, which determines the scale at which \n" \
                                           "embedded points will be spread out")
        if str(Event.widget) == ".!frame9.!button8":  # spread
            LabelingUMAPDialog.config(text="The effective scale of embedded points. In combination with \n" \
                                           "\"min_dist\" this determines how clustered/clumped the embedded points are")
        if str(Event.widget) == ".!frame9.!button10":  # metric
            LabelingUMAPDialog.config(text="The metric to use to compute distances in high dimensional space")
        if str(Event.widget) == ".!frame9.!button12":  # random_state
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
    '''

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
    BirdIDText.grid(row=2, column=1, sticky="w")

    BirdIDLabel = tk.Label(SegmentationMainFrame, text="Bird ID:")
    BirdIDLabel.grid(row=2, column=0)

    FileExplorer = tk.Button(SegmentationMainFrame, text="Find Folder", command=lambda: FileExplorerFunction())
    FileExplorer.grid(row=1, column=0)

    SegmentButton = tk.Button(SegmentationMainFrame, text="Segment!", command=lambda: SegmentButtonClick())
    SegmentButton.grid(row=6, columnspan=2)

    
    ttk.Style().theme_use("clam")
    gui.mainloop()

except Exception:
    print(Exception)