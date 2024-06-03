######  Required libraries to install: ########
# avn
# pymde
# hdbscan
# Microsoft Visual C++ 14.0 or greater (required for hdbscan)
#           https://visualstudio.microsoft.com/visual-cpp-build-tools/


#Upgrade seaborn from 0.11.2 to 0.12.2
# pandas is version 2.0.3

# To export gui to .exe file, open environment, navigate to gui.py directory and type "pyinstaller --collect-submodules "sklearn" gui.py"


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
#import customtkinter ### Causes an issue for Windows 11
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
import webbrowser as wb
from tkinter.filedialog import asksaveasfile
import shutil
import avn.similarity as similarity
from sklearn.decomposition import PCA
import datetime
from datetime import date

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
    global SyntaxAlignmentVar
    global AlignmentChoices

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
        if Type == "Songs":
            global LabelingSongLocation
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                LabelingSongLocation.config(text=song_path_temp)
                LabelingSongLocation.update()
    elif Module == "Labeling_UMAP":
        if Type == "Input":
            global LabelingSpectrogramFiles
            global LabelingFileDisplay
            LabelingFile = filedialog.askopenfilenames(filetypes=[(".csv files", "*.csv")])
            if len(LabelingFile) > 0:
                LabelingSpectrogramFiles = LabelingFile
                LabelingFileDisplay.config(text=LabelingFile)
                LabelingFileDisplay.update()
                # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                if pattern.search(LabelingFile[0]) != None:
                    global BirdID_Labeling
                    BirdID_Labeling.delete(0, tk.END)
                    BirdID_Labeling.config(fg='black')
                    BirdID_Labeling.insert(0, str(pattern.search(LabelingFile[0]).group()))
                    BirdID_Labeling.update()
        if Type == "Output":
            global LabelingUMAP_OutputText
            OutputDir = filedialog.askdirectory()
            if len(OutputDir) > 0:
                LabelingUMAP_OutputText.config(text=OutputDir)
                LabelingUMAP_OutputText.update()
        if Type == "Songs":
            global FindSongPath
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                FindSongPath.config(text=song_path_temp)
                FindSongPath.update()
    elif Module == "Labeling_Bulk":
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
    elif Module == "Timing":
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
        if Type == "Songs":
            global TimingSongPath
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                TimingSongPath.config(text=song_path_temp)
                TimingSongPath.update()
    elif Module == "Timing_Rhythm":
        if Type == "Input":
            TimingInput = filedialog.askdirectory()
            if len(TimingInput) > 0:
                RhythmSpectrogram_Input.config(text=TimingInput)
                RhythmSpectrogram_Input.update()
                global RhythmSpectrogram_BirdID
                # print(str(glob.glob(TimingInput+"/*.wav")[0]))
                if pattern.search(TimingInput) != None:
                    RhythmSpectrogram_BirdID.delete(0, tk.END)
                    RhythmSpectrogram_BirdID.config(fg='black')
                    RhythmSpectrogram_BirdID.insert(0, str(pattern.search(TimingInput).group()))
                    RhythmSpectrogram_BirdID.update()
                elif pattern.search(str(glob.glob(TimingInput+"/*.wav")[0])) != None:
                    RhythmSpectrogram_BirdID.delete(0, tk.END)
                    RhythmSpectrogram_BirdID.config(fg='black')
                    RhythmSpectrogram_BirdID.insert(0, str(pattern.search(str(glob.glob(TimingInput+"/*.wav")[0])))[:-2].split("match=\'")[1])
                    RhythmSpectrogram_BirdID.update()
        if Type == "Output":
            TimingOutput = filedialog.askdirectory()
            if len(TimingOutput) > 0:
                RhythmSpectrogram_Output.config(text=TimingOutput)
                RhythmSpectrogram_Output.update()
    elif Module == "Acoustics_Single":
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
                    # global AcousticsFileDuration
                    # AcousticsFileDuration = tk.Label(AcousticsMainFrameSingle, text="          Duration: " + str(
                    #     audioread.audio_open(AcousticsDirectory).duration) + " sec", justify="center")
                    # AcousticsFileDuration.grid(row=0, column=3)
                    ResetAcousticsOffset()
                    # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                    # if pattern.search(AcousticsDirectory) != None:
                    #     BirdIDText.delete(0, tk.END)
                    #     BirdIDText.config(fg='black')
                    #     BirdIDText.insert(0, str(pattern.search(AcousticsDirectory).group()))
        if Type == "Output":
            global AcousticsOutput_Text
            OutputDir = filedialog.askdirectory()
            if len(OutputDir) > 0:
                AcousticsOutput_Text.config(text=OutputDir)
                AcousticsOutput_Text.update()
    elif Module == "Acoustics_Multi":
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
                if pattern.search(MultiAcousticsDirectory) != None:
                    MultiAcousticsBirdID.delete(0, tk.END)
                    MultiAcousticsBirdID.config(fg='black')
                    MultiAcousticsBirdID.insert(0, str(pattern.search(MultiAcousticsDirectory).group()))
        if Type == "Output":
            global MultiAcousticsOutputDisplay
            OutputDir = filedialog.askdirectory()
            if len(OutputDir) > 0:
                MultiAcousticsOutputDisplay.config(text=OutputDir)
                MultiAcousticsOutputDisplay.update()
        if Type == "Songs":
            global MultiAcousticsSongPath
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                MultiAcousticsSongPath.config(text=song_path_temp)
                MultiAcousticsSongPath.update()
    elif Module == "Syntax":
        if Type == "Input":
            global SyntaxDirectory
            # global AlignmentChoices
            global SyntaxAlignment
            SyntaxDirectory_temp = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
            if len(SyntaxDirectory_temp) > 0:
                global SyntaxFileDisplay
                SyntaxFileDisplay.config(text=SyntaxDirectory_temp)
                SyntaxFileDisplay.update()

                if pattern.search(SyntaxDirectory_temp.split("/")[-1]) != None:
                    SyntaxBirdID.delete(0, tk.END)
                    SyntaxBirdID.config(fg='black')
                    SyntaxBirdID.insert(0, str(pattern.search(SyntaxDirectory_temp).group()))

                SyntaxDirectory = SyntaxDirectory_temp
                labeling_data = pd.read_csv(SyntaxDirectory)
                for i in labeling_data["labels"].unique():
                    AlignmentChoices.append(str(i))
                AlignmentChoices.pop(0)
                AlignmentChoices.sort()
                AlignmentChoices.insert(0, "Auto")
                SyntaxAlignmentVar.set("Auto")
                SyntaxAlignment.destroy()
                SyntaxAlignment = tk.OptionMenu(Syntax_RasterTab, SyntaxAlignmentVar, *AlignmentChoices)
                SyntaxAlignment.grid(row=4, column=1, sticky="w", padx=Padding_Width)
        if Type == "Output":
            global SyntaxOutputDisplay
            OutputDir = filedialog.askdirectory()
            if len(OutputDir) > 0:
                SyntaxOutputDisplay.config(text=OutputDir)
                SyntaxOutputDisplay.update()
        if Type == "Songs":
            global SyntaxSongLocation
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                SyntaxSongLocation.config(text=song_path_temp)
                SyntaxSongLocation.update()
    elif Module == "Plain_Folder":
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
                PlainOutputFolder_Label.config(text=PlainOutputDir_temp)
                PlainOutputFolder_Label.update()
    elif Module == "Plain_Files":
        if Type == "Input":
            global PlainDirectoryAlt
            PlainDirectoryAlt_temp = filedialog.askopenfilenames(filetypes=[(".wav files", "*.wav")])
            if len(PlainDirectoryAlt_temp) > 0:
                if len(PlainDirectoryAlt_temp) == 1:
                    PlainDirectoryLabelAlt.config(text=str(len(PlainDirectoryAlt_temp)) + " file selected")
                else:
                    PlainDirectoryLabelAlt.config(text=str(len(PlainDirectoryAlt_temp)) + " files selected")
                PlainDirectoryLabelAlt.update()
            PlainDirectoryAlt = PlainDirectoryAlt_temp
        if Type == "Output":
            global PlainOutputAlt_Label
            PlainOutputAlt_temp = filedialog.askdirectory()
            if len(PlainOutputAlt_temp) > 0:
                PlainOutputAlt_Label.config(text=PlainOutputAlt_temp)
                PlainOutputAlt_Label.update()
    elif Module == "Labeling_Extra":
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
    elif Module == "Similarity_Prep":
        if Type == "Input":
            global SimilarityInput
            InputDir_temp = filedialog.askopenfilenames(filetypes=[(".csv files", "*.csv")])
            if len(InputDir_temp) > 0:
                SimilarityInput.config(text=InputDir_temp)
                SimilarityInput.update()
            global Similarity_BirdID
            if pattern.search(str(InputDir_temp)) != None:
                Similarity_BirdID.delete(0, tk.END)
                Similarity_BirdID.config(fg='black')
                Similarity_BirdID.insert(0, str(pattern.search(str(InputDir_temp)).group()))
        if Type == "Output":
            global SimilarityOutput
            OutputDir_temp = filedialog.askdirectory()
            if len(OutputDir_temp) > 0:
                SimilarityOutput.config(text=OutputDir_temp)
                SimilarityOutput.update()
        if Type == "Songs":
            global SimilaritySongPath
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                SimilaritySongPath.config(text=song_path_temp)
                SimilaritySongPath.update()
    elif Module == "Similarity_Out1":
        if Type == "Input":
            global SimilarityInput2
            InputDir_temp = filedialog.askdirectory()
            if len(InputDir_temp) > 0:
                SimilarityInput2.config(text=InputDir_temp)
                SimilarityInput2.update()
            global Similarity_BirdID2
            if pattern.search(InputDir_temp) != None:
                Similarity_BirdID2.delete(0, tk.END)
                Similarity_BirdID2.config(fg='black')
                Similarity_BirdID2.insert(0, str(pattern.search(InputDir_temp).group()))
        if Type == "Output":
            global SimilarityOutput2
            OutputDir_temp = filedialog.askdirectory()
            if len(OutputDir_temp) > 0:
                SimilarityOutput2.config(text=OutputDir_temp)
                SimilarityOutput2.update()
    elif Module == "Similarity_Out2":
        if Type == "Input":
            global SimilarityInput3
            InputDir_temp = filedialog.askdirectory()
            if len(InputDir_temp) > 0:
                SimilarityInput3.config(text=InputDir_temp)
                SimilarityInput3.update()
            global Similarity_BirdID3
            if pattern.search(InputDir_temp) != None:
                Similarity_BirdID3.delete(0, tk.END)
                Similarity_BirdID3.config(fg='black')
                Similarity_BirdID3.insert(0, str(pattern.search(InputDir_temp).group()))
        if Type == "Output":
            global SimilarityOutput3
            OutputDir_temp = filedialog.askdirectory()
            if len(OutputDir_temp) > 0:
                SimilarityOutput3.config(text=OutputDir_temp)
                SimilarityOutput3.update()
    elif Module == "RunAll":
        if Type == "Input":
            RunAllInput = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
            if len(RunAllInput) > 0:
                global RunAll_InputFileDisplay
                RunAll_InputFileDisplay.config(text=RunAllInput)
                RunAll_InputFileDisplay.update()
                if pattern.search(RunAllInput) != None:
                    RunAll_BirdID_Entry.delete(0, tk.END)
                    RunAll_BirdID_Entry.config(fg='black')
                    RunAll_BirdID_Entry.insert(0, str(pattern.search(RunAllInput).group()))
                    RunAll_BirdID_Entry.update()
        if Type == "Output":
            RunAllOutput = filedialog.askdirectory()
            if len(RunAllOutput) > 0:
                global RunAll_OutputFileDisplay
                RunAll_OutputFileDisplay.config(text=RunAllOutput)
                RunAll_OutputFileDisplay.update()
        if Type == "Songs":
            global RunAll_SongPath
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                RunAll_SongPath.config(text=song_path_temp)
                RunAll_SongPath.update()
    elif Module == "SyntaxHeatmap":
        if Type == "Input":
            Input = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
            if len(Input) > 0:
                global Syntax_HeatmapTab_Input
                Syntax_HeatmapTab_Input.config(text=Input)
                Syntax_HeatmapTab_Input.update()
                if pattern.search(Input) != None:
                    Syntax_HeatmapTab_BirdID.delete(0, tk.END)
                    Syntax_HeatmapTab_BirdID.config(fg='black')
                    Syntax_HeatmapTab_BirdID.insert(0, str(pattern.search(Input).group()))
                    Syntax_HeatmapTab_BirdID.update()
        if Type == "Output":
            Output = filedialog.askdirectory()
            if len(Output) > 0:
                global Syntax_HeatmapTab_Output
                Syntax_HeatmapTab_Output.config(text=Output)
                Syntax_HeatmapTab_Output.update()
        if Type == "Song":
            Input = filedialog.askdirectory()
            if len(Input) > 0:
                global Syntax_HeatmapTab_SongLocation
                Syntax_HeatmapTab_SongLocation.config(text=Input)
                Syntax_HeatmapTab_SongLocation.update()
    elif Module == "SyntaxRaster":
        if Type == "Input":
            Input = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
            if len(Input) > 0:
                global Syntax_Raster_Input
                Syntax_Raster_Input.config(text=Input)
                Syntax_Raster_Input.update()
                if pattern.search(Input) != None:
                    Syntax_Raster_BirdID.delete(0, tk.END)
                    Syntax_Raster_BirdID.config(fg='black')
                    Syntax_Raster_BirdID.insert(0, str(pattern.search(Input).group()))
                    Syntax_Raster_BirdID.update()
                AlignmentChoices_temp = pd.read_csv(Input)["labels"].unique()
                AlignmentChoices = []
                for x in AlignmentChoices_temp:
                    AlignmentChoices.append(x)
                try:
                    AlignmentChoices.remove("file_start")
                except: pass
                try:
                    AlignmentChoices.remove("silent_gap")
                except: pass
                try:
                    AlignmentChoices.remove("file_end")
                except: pass
                AlignmentChoices.sort()
                AlignmentChoices.insert(0, "Auto")
                SyntaxAlignment.destroy()
                SyntaxAlignmentVar.set("Auto")
                SyntaxAlignment = tk.OptionMenu(Syntax_RasterTab, SyntaxAlignmentVar, *AlignmentChoices)
                SyntaxAlignment.grid(row=4, column=1, sticky="w", padx=Padding_Width)

        if Type == "Output":
            Output = filedialog.askdirectory()
            if len(Output) > 0:
                global Syntax_Raster_Output
                Syntax_Raster_Output.config(text=Output)
                Syntax_Raster_Output.update()

        if Type == "Song":
            Input = filedialog.askdirectory()
            if len(Input) > 0:
                global Syntax_Raster_SongLocation
                Syntax_Raster_SongLocation.config(text=Input)
                Syntax_Raster_SongLocation.update()
    elif Module == "GlobalInputs":
        if Type == "Input":
            InputTemp = filedialog.askopenfilename(filetypes=[(".csv file", "*.csv")])
            if len(InputTemp) > 0:
                if len(InputTemp) > 0:
                    Input = InputTemp
                    GlobalInputs_Input.config(text=Input)
                    GlobalInputs_Input.update()
                    # Automatically finds Bird ID from file path, if possible, and enters into Bird ID field
                    if pattern.search(Input) != None:
                        GlobalInputs_BirdID.delete(0, tk.END)
                        GlobalInputs_BirdID.config(fg='black')
                        GlobalInputs_BirdID.insert(0, str(pattern.search(Input).group()))
                        GlobalInputs_BirdID.update()
                        #Bird_ID = str(pattern.search(segmentations_path).group())
        if Type == "Output":
            Selected_Output = filedialog.askdirectory()
            if len(Selected_Output) > 0:
                GlobalInputs_Output.config(text=str(Selected_Output))
                GlobalInputs_Output.update()
        if Type == "Songs":
            song_path_temp = filedialog.askdirectory()
            if len(song_path_temp) > 0:
                GlobalInputs_SongDir.config(text=song_path_temp)
                GlobalInputs_SongDir.update()

def ResetAcousticsOffset():
    global AcousticsInputVar
    AcousticsOffset.delete(0,END)
    if AcousticsInputVar.get() == 0:
        AcousticsOffset.insert(0, audioread.audio_open(AcousticsDirectory).duration)
    if AcousticsInputVar.get() == 1:
        AcousticsOffset.insert(0, "End of File")
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
    global LabelingSongLocation
    Song_Location = LabelingSongLocation.cget("text")
    if Song_Location[-1] != "/" or Song_Location[-1] != "\\":
        Song_Location = Song_Location+"/"
    UMAP_Directories = []
    Bird_ID = LabelingBirdIDText.get()

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
                global EndOfFolder

                LabelingProgress_Text = tk.Label(LabelingMainFrame, text="", justify="center")
                LabelingProgress_Text.grid(row=9, column=0, columnspan=2)
                gui.update_idletasks()
                OutputFolder = LabelingOutputFile_Text.cget("text")

                for directory in segmentations_path:

                    directory = directory.replace("\\", "/")

                    # dir_corrected = directory.split("/")[-1].replace("_seg_", "_label_")
                    # dir_corrected = dir_corrected.replace("_wseg.csv", "_labels.csv")
                    try:
                        os.makedirs(OutputFolder+"/Labeling")
                    except: pass
                    output_file = OutputFolder + "/Labeling/" + Bird_ID + "_labels.csv"  # e.g. "C:/where_I_want_the_labels_to_be_saved/Bird_ID_labels.csv"

                    #############################################
                    # This is from labeling tutorial:
                    def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
                        spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length,
                                                   n_fft=n_fft)
                        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

                        # normalize
                        S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

                        return S_norm

                    # hop_length = int(hop_length_entry_Labeling.get())
                    # win_length = int(win_length_entry_Labeling.get())
                    # n_fft = int(n_fft_entry_Labeling.get())
                    # ref_db = int(ref_db_entry_Labeling.get())
                    # amin = float(a_min_entry_Labeling.get())
                    # min_level_db = int(min_level_db_entry_Labeling.get())

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
                        file_path = Song_Location.replace("\\", "/") + song_file

                        song = dataloading.SongFile(file_path)
                        song.bandpass_filter(int(bandpass_lower_cutoff_entry_Labeling.get()),
                                             int(bandpass_upper_cutoff_entry_Labeling.get()))

                        syllable_df = segmentations[segmentations['files'] == song_file]
                        # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
                        syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                                for st, et in
                                                zip(syllable_df.onsets.values, syllable_df.offsets.values)]
                        syllable_dfs = pd.concat([syllable_dfs, syllable_df])

                    LabelingProgress.set(0)
                    # Normalize the audio  --- Ethan's comment: This won't work when there's an empty array for syllable_dfs_audio_values, so I'm just going to set those to '[0]'

                    syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
                    global hdbscan_df
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

                            # if syllable_spec.shape[1] > int(max_spec_size_entry_Labeling.get()):
                            #     print(
                            #         "Long Syllable Corrections! Spectrogram Duration = " + str(
                            #             syllable_spec.shape[1]))
                            #     syllable_spec = syllable_spec[:, :int(max_spec_size_entry_Labeling.get())]

                            syllables_spec.append(syllable_spec)

                    LabelingProgress_Text.config(text="Processing...")
                    LabelingProgress_Text.update()
                    LabelingProgress_Bar.config(mode="indeterminate")
                    LabelingProgress_Bar.start()
                    LabelingProgress_Bar.update()
                    time.sleep(0.1)
                    gui.update_idletasks()
                    time.sleep(0.1)

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
                    mde = pymde.preserve_neighbors(specs_flattened_array, n_neighbors=int(n_neighbors_entry_Labeling.get()),
                                                   embedding_dim=int(n_components_entry_Labeling.get()))
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

                    for column in hdbscan_df.columns:
                        if "Unnamed" in column:
                            hdbscan_df = hdbscan_df.drop(column, axis=1)

                    hdbscan_df.to_csv(output_file)
                    # os.rename(output_file+Bird_ID+"_labels.csv", output_file+Bird_ID+"_labels_"+output_file.split("/")[-1]+".csv")
                    # ------------------------------------

                    LabelingProgress_Text.config(text="Saving...")
                    LabelingProgress_Text.update()

                    # Create UMAP
                    plt.clf()
                    plt.figure(figsize=(5,5))
                    LabelingScatterplot = sns.scatterplot(data=hdbscan_df, x="X", y="Y", hue="labels", alpha=0.25,
                                                          s=5)
                    plt.title(str(Bird_ID));
                    sns.move_legend(LabelingScatterplot, "upper right")
                    UMAP_Fig = LabelingScatterplot.get_figure()

                    # song_folder = song_folder_dir
                    dir_corrected = dir_corrected.replace("_wseg.csv", "_labels.csv")
                    UMAP_Output = output_file.replace("_label_", "_UMAP_")
                    if "_labels.csv" in UMAP_Output:
                        UMAP_Output = UMAP_Output.replace("_labels.csv", "_UMAP-Clusters.png")
                    else:
                        UMAP_Output = UMAP_Output.replace(".csv", "_UMAP-Clusters.png")
                    UMAP_Fig.savefig(UMAP_Output)
                    plt.clf()

                    UMAP_Directories.append(UMAP_Output)

                    # Generate Metadata File for Advanced Settings #
                    LabelingSettingsMetadata = pd.DataFrame({
                                                            "date":[str(date.today().strftime("%m/%d/%y"))],
                                                            "avn version":["0.5.1"],
                                                            "gui version":["0.1.0"],
                                                            "Bird_ID":[str(Bird_ID)],
                                                            "bandpass_lower_cutoff":[bandpass_lower_cutoff_entry_Labeling.get()], # Labeling Settings
                                                            "bandpass_upper_cutoff":[bandpass_upper_cutoff_entry_Labeling.get()],
                                                            "a_min":[a_min_entry_Labeling.get()],
                                                            "ref_db":[ref_db_entry_Labeling.get()],
                                                            "min_level_db":[min_level_db_entry_Labeling.get()],
                                                            "n_fft":[n_fft_entry_Labeling.get()],
                                                            "win_length":[win_length_entry_Labeling.get()],
                                                            "hop_length":[hop_length_entry_Labeling.get()],
                                                            'n_neighbors':[n_neighbors_entry_Labeling.get()], # UMAP Settings
                                                            'n_components':[n_components_entry_Labeling.get()],
                                                            'min_dist':[min_dist_entry_Labeling.get()],
                                                            'spread':[spread_entry_Labeling.get()],
                                                            'metric':[metric_variable.get()],
                                                            'random_state':[random_state_entry_Labeling.get()],
                                                            'min_cluster_prop':[min_cluster_prop_entry_Labeling.get()], # Clustering Settings
                                                            'min_samples':[min_samples_entry_Labeling.get()]
                                                            })
                    NewMetaCols = [('date','date'),("avn version","avn version"),("gui version","gui version"), ("Bird_ID","Bird_ID"),
                                    ('Labeling',"bandpass_lower_cutoff"),('Labeling',"bandpass_upper_cutoff"),
                                   ('Labeling',"a_min"),('Labeling',"ref_db"),
                                   ('Labeling',"min_level_db"),('Labeling',"n_fft"),
                                   ('Labeling',"win_length"),('Labeling',"hop_length"),
                                   ('UMAP',"n_neighbors"),('UMAP',"n_components"),
                                   ('UMAP',"min_dist"),('UMAP',"spread"),
                                   ('UMAP',"metric"),('UMAP',"random_state"),
                                   ('Clustering','min_cluster_prop'),('Clustering','min_samples')]

                    LabelingSettingsMetadata.columns = pd.MultiIndex.from_tuples(NewMetaCols)

                    Meta_Output = UMAP_Output.replace("_UMAP-Clusters.png", "_Labeling_Metadata.csv")
                    LabelingSettingsMetadata.to_csv(Meta_Output)

                    LabelingProgress_Bar.destroy()
                    LabelingProgress_Text.config(text="Labeling Complete!")
                    LabelingProgress_Text.update()
                    time.sleep(3)
                    LabelingProgress_Text.destroy()

                    # Make folder for storing labeling photos

                    def UMAP_Display():
                        global UMAP_Directories

                        UMAP_Img = PhotoImage(file=UMAP_Directories[0])
                        try:
                            UMAP_ImgLabel.destroy()
                        except:
                            pass
                        def UMAP_Save():
                            global LabelingFig
                            global UMAP_Directories
                            FileName_temp=UMAP_Directories[0].replace("\\","/").split("/")
                            FileName=""
                            for i in FileName_temp[:-1]:
                                FileName = FileName+i+"/"
                            file = asksaveasfile(initialfile=str(FileName) + LabelingBirdIDText.get()+'_UMAP.png',
                                                 defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
                            try:
                                shutil.rmtree(file)
                            except:
                                pass
                            filename = str(file).split("\'")[1]
                            LabelingFig.savefig(filename)
                        UMAP_SaveButton = tk.Button(UMAPWindow, text="Save",command=lambda: UMAP_Save())
                        UMAP_SaveButton.grid(row=0, column=0)
                        UMAP_ImgLabel = tk.Label(UMAPWindow, image=UMAP_Img)
                        UMAP_ImgLabel.grid(row=1, columnspan=3)
                        UMAP_ImgLabel.update()

                        UMAPWindow = tk.Toplevel(gui)

                        UMAPWindow.mainloop()


                    EndOfFolder = False
                    LabelingDisplay("Start")
                    UMAP_Display()



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
        Bird_ID = LabelingBirdIDText.get()
        label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(syll_df=hdbscan_df, song_folder_path=hdbscan_df["directory"][1], Bird_ID="",song_file_index=0,figsize=(12, 4), fontsize=14)


        LabelingDisplayFrame = tk.Frame(LabelingDisplayWindow)
        LabelingDisplayFrame.grid(row=1, column=0, columnspan=21)
        label_canvas = FigureCanvasTkAgg(label_fig, master=LabelingDisplayFrame)  # A tk.DrawingArea.
        label_canvas.draw()
        label_canvas.get_tk_widget().grid()
    if Direction == "Left":
        if label_FileID != 0:
            label_FileID -= 1

            label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, LabelingSongLocation.cget("text"), "",
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

            label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, LabelingSongLocation.cget("text"), "",
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

    file = asksaveasfile(initialfile=str(FileName)+'_LabeledSpectrogram.png',defaultextension=".png", filetypes=[("PNG Image", "*.png"),("All Files", "*.*")])
    try:
        shutil.rmtree(file)
    except:
        pass
    filename = str(file).split("\'")[1]
    label_fig.savefig(filename)

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
        # try:
        #     os.makedirs(song_folder+"/LabelingPhotos")
        # except: pass
        if LabelingSaveAllCheck.get() == 1:
            i = glob.glob(str(song_folder) + "/*.wav")
            for a in range(len(i)):
                LabelingSaveProgress.config(
                    text="Saving Labeling Images (" + str(a + 1) + " of " + str(len(i)) + ")")
                LabelingSaveProgress.update()
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                                    song_file_index=a, figsize=(20, 5),
                                                                                    fontsize=14)
                fig.savefig(str(song_folder) + str(song_file_name) + ".png")
        if LabelingSaveAllCheck.get() == 0:
            for a in BulkLabelFiles_to_save:
                LabelingSaveProgress.config(
                    text="Saving Labeling Images (" + str(BulkLabelFiles_to_save.index(a) + 1) + " of " + str(len(BulkLabelFiles_to_save)) + ")")
                LabelingSaveProgress.update()
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, song_folder, "",
                                                                                    song_file_index=BulkLabelFiles_to_save.index(a), figsize=(20, 5),
                                                                                    fontsize=14)
                fig.savefig(str(song_folder) + str(song_file_name) + ".png")
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

        hop_length = int(hop_length_entry_Labeling.get())
        win_length = int(win_length_entry_Labeling.get())
        n_fft = int(n_fft_entry_Labeling.get())
        ref_db = int(ref_db_entry_Labeling.get())
        amin = float(a_min_entry_Labeling.get())
        min_level_db = int(min_level_db_entry_Labeling.get())

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
        global LabelingSongLocation
        for song_file in segmentations.files.unique():

            SongIndex = segmentations.index[segmentations['files'] == song_file].tolist()
            file_path = LabelingSongLocation + "/" + song_file

            song = dataloading.SongFile(file_path)
            song.bandpass_filter(int(bandpass_lower_cutoff_entry_Labeling.get()),
                                 int(bandpass_upper_cutoff_entry_Labeling.get()))

            syllable_df = segmentations[segmentations['files'] == song_file]
            # this section is based on avn.signalprocessing.create_spectrogram_dataset.get_row_audio()
            syllable_df["audio"] = [song.data[int(st * song.sample_rate): int(et * song.sample_rate)]
                                    for st, et in
                                    zip(syllable_df.onsets.values, syllable_df.offsets.values)]
            syllable_dfs = pd.concat([syllable_dfs, syllable_df])

        syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
        global hdbscan_df
        hdbscan_df = syllable_dfs

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
            # if syllable_spec.shape[1] > int(max_spec_size_entry_Labeling.get()):
            #     print(
            #         "Long Syllable Corrections! Spectrogram Duration = " + str(
            #             syllable_spec.shape[1]))
            #     syllable_spec = syllable_spec[:, :int(max_spec_size_entry_Labeling.get())]

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
    global MaxFiles_Labeling



    # LabelingFile = LabelingFileDisplay.cget("text")
    global FindSongPath
    SongPath = FindSongPath.cget("text")+"/"
    SongPath = SongPath.replace("\\","/")

    try:
        LabelingSpectrogram_LoadingMessage.destroy()
    except: pass
    try:
        LabelingSpectrogram_LoadingBar.destroy()
    except: pass
    LoadingVar = IntVar()
    LabelingSpectrogram_LoadingBar = ttk.Progressbar(LabelingSpectrogram_UMAP, mode="determinate",
                                                     maximum=3, variable=LoadingVar)
    LabelingSpectrogram_LoadingBar.grid(row=10, column=0, columnspan=2)
    LabelingSpectrogram_LoadingMessage = tk.Label(LabelingSpectrogram_UMAP, text="Loading Data...")
    LabelingSpectrogram_LoadingMessage.grid(row=9, column=0, columnspan=2)
    def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
        spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length,
                                   n_fft=n_fft)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

        # normalize
        S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

        return S_norm

    LabelingSpectrogram_LoadingBar.step(1)
    LabelingSpectrogram_LoadingBar.update()

    hop_length = int(hop_length_entry_Labeling.get())
    win_length = int(win_length_entry_Labeling.get())
    n_fft = int(n_fft_entry_Labeling.get())
    ref_db = int(ref_db_entry_Labeling.get())
    amin = float(a_min_entry_Labeling.get())
    min_level_db = int(min_level_db_entry_Labeling.get())

    K = 10
    min_cluster_prop = 0.04
    embedding_dim = 2
    LabelingSpectrogram_LoadingBar.step(1)
    LabelingSpectrogram_LoadingBar.update()
    # load segmentations
    global LabelingFileDisplay
    segmentations_temp = pd.read_csv(LabelingFileDisplay.cget("text"))
    try:
        segmentations_temp = segmentations_temp.rename(
        columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
    except:
        pass
    UniqueFiles_temp = pd.unique(segmentations_temp["files"])
    FilesToCheck = min(int(MaxFiles_Labeling.get()), len(UniqueFiles_temp))
    UniqueFiles = UniqueFiles_temp[:FilesToCheck]
    segmentations = segmentations_temp[segmentations_temp["files"].isin(UniqueFiles)]

    # sometimes this padding can create negative onset times, which will cause errors.
    # correct this by setting any onsets <0 to 0.
    segmentations.onsets = segmentations.where(segmentations.onsets > 0, 0).onsets

    # Add syllable audio to dataframe
    syllable_dfs = pd.DataFrame()
    # for directory in segmentations.directory.unique():

    LabelingSpectrogram_LoadingBar.step(1)
    LabelingSpectrogram_LoadingBar.update()

    LabelingSpectrogram_LoadingMessage.config(text="Processing Data...")
    LabelingSpectrogram_LoadingMessage.update()
    LabelingSpectrogram_LoadingBar.config(maximum=len(segmentations.files.unique()))
    LoadingVar.set(0)
    LabelingSpectrogram_LoadingBar.update()

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
        syllable_dfs = pd.concat([syllable_dfs, syllable_df])
        LabelingSpectrogram_LoadingBar.step(1)

    LabelingSpectrogram_LoadingMessage.config(text="Generating Spectrograms...")
    LabelingSpectrogram_LoadingMessage.update()
    LabelingSpectrogram_LoadingBar.config(maximum=len(syllable_dfs.audio.values))
    LoadingVar.set(0)
    LabelingSpectrogram_LoadingBar.update()

    syllable_dfs['audio'] = [librosa.util.normalize(i) for i in syllable_dfs.audio.values]
    # global hdbscan_df
    hdbscan_df = syllable_dfs

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
        syllables_spec.append(syllable_spec)
        LabelingSpectrogram_LoadingBar.step(1)
        LabelingSpectrogram_LoadingBar.update()

    LabelingSpectrogram_LoadingMessage.config(text="Saving Spectrograms...")
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


    # output_file = OutputFolder + "/Labeling_temp/" + dir_corrected
    # hdbscan_df.to_csv(output_file)
    i_end = min(int(MaxFiles_Labeling.get()),len(hdbscan_df["files"].unique()))

    LabelingSpectrogram_LoadingBar.config(maximum=i_end)
    LoadingVar.set(0)
    LabelingSpectrogram_LoadingBar.update()

    try:
        os.makedirs(LabelingUMAP_OutputText.cget("text")+"/LabeledSpectrograms/")
    except:
        pass
    for i in range(i_end):
        LabelingSpectrogram_LoadingMessage.config(text="Saving file "+str(i+1)+" of "+str(i_end)+"...")
        label_fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(hdbscan_df, FindSongPath.cget('text'),
                                                                              "",
                                                                              song_file_index=i,
                                                                              figsize=(12, 4), fontsize=14)
        label_fig.savefig(LabelingUMAP_OutputText.cget("text")+"/LabeledSpectrograms/"+str(song_file_name)[:-4]+".png")
        LabelingSpectrogram_LoadingBar.step(1)
        LabelingSpectrogram_LoadingBar.update()
    LabelingSpectrogram_LoadingMessage.config(text="Spectrograms Saved!")
    LabelingSpectrogram_LoadingBar.destroy()
    gui.update_idletasks()
    def OldStuff():
        for file in LabelingSpectrogramFiles:
            LabelingData = pd.read_csv(file)
            Split_LabelingFile = file.split("/")[:-1]
            Merged_LabelingFile = ""
            for a in Split_LabelingFile:
                Merged_LabelingFile = Merged_LabelingFile + a + "/"
            global LabelingUMAP_OutputText
            LabelingOutputDir = LabelingUMAP_OutputText.cget('text')+"/"

            global MaxFiles_Labeling
            MaxFiles = int(MaxFiles_Labeling.get())
            if len(LabelingData["files"]) < MaxFiles:
                i_end = len(LabelingData["files"])
            else:
                i_end = MaxFiles
            LabelingSpectrogram_LoadingBar = ttk.Progressbar(LabelingSpectrogram_UMAP, mode="determinate",
                                                             maximum=i_end)
            LabelingSpectrogram_LoadingBar.grid(row=10, column=0, columnspan=2)
            LabelingSpectrogram_LoadingMessage = tk.Label(LabelingSpectrogram_UMAP, text="")
            LabelingSpectrogram_LoadingMessage.grid(row=9, column=0, columnspan=2)
            for i in range(i_end):
                LabelingSpectrogram_LoadingMessage.config(
                    text="Processing file " + str(i+1) + " of " + str(i_end))
                LabelingSpectrogram_LoadingBar.step(1)
                fig, ax, song_file_name = avn.plotting.plot_spectrogram_with_labels(LabelingData,
                                                                                    SongPath.replace("\\", ""), "",
                                                                                    song_file_index=i,
                                                                                    figsize=(15, 4),
                                                                                    fontsize=14)
                try:
                    shutil.rmtree(LabelingOutputDir + str(song_file_name) + ".png")
                except:
                    pass
                song_file_name = song_file_name[:-4]
                fig.savefig(LabelingOutputDir + str(song_file_name) + ".png")
                plt.close(fig)
                time.sleep(.1)
                gui.update_idletasks()
            LabelingSpectrogram_LoadingBar.destroy()
            time.sleep(.1)
            gui.update_idletasks()
            LabelingSpectrogram_LoadingMessage.config(text="Spectrogram Generation Complete!")
            time.sleep(3)
            LabelingSpectrogram_LoadingMessage.destroy()
            gui.update_idletasks()

def Acoustics_Interval():
    global ContCalc
    global AcousticsProgress
    try:
        AcousticsProgress.config(text="Calculating Acoustics...")
        AcousticsProgress.update()
    except:
        AcousticsProgress = tk.Label(AcousticsMainFrameSingle, text="Calculating Acoustics...")
        AcousticsProgress.grid(row=20, column=1, columnspan=2)

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

    global AcousticsInputVar
    global AcousticsOutput_Text
    if AcousticsInputVar.get() == 0: # Process a single file
        song = dataloading.SongFile(AcousticsDirectory)
        if AcousticsOffset.get() == "End of File":
            song_interval = acoustics.SongInterval(song, onset=float(AcousticsOnset.get()), offset=None,
                                                   win_length=int(win_length_entry_Acoustics.get()),
                                                   hop_length=int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()),
                                                   max_F0=int(max_F0_entry_Acoustics.get()),
                                                   min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=float(freq_range_entry_Acoustics.get()),
                                                   baseline_amp=int(baseline_amp_entry_Acoustics.get()),
                                                   fmax_yin=int(fmax_yin_entry_Acoustics.get()))
        else:
            song_interval = acoustics.SongInterval(song, onset=float(AcousticsOnset.get()),
                                                   offset=float(AcousticsOffset.get()),
                                                   win_length=int(win_length_entry_Acoustics.get()),
                                                   hop_length=int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()),
                                                   max_F0=int(max_F0_entry_Acoustics.get()),
                                                   min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=float(freq_range_entry_Acoustics.get()),
                                                   baseline_amp=int(baseline_amp_entry_Acoustics.get()),
                                                   fmax_yin=int(fmax_yin_entry_Acoustics.get()))
        #song_interval.calc_all_feature_stats(features=FeatureList)
        # song_interval.save_features(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/",file_name=str(AcousticsDirectory.split("/")[-1][:-4]))
        #feature_stats = song_interval.calc_feature_stats(features=FeatureList)
        if ContCalc.get() == 0: # Get values for each time bin
            song_interval.save_features(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/",file_name=str(AcousticsDirectory.split("/")[-1][:-4]), features=FeatureList)
        if ContCalc.get() == 1: # Get average values across time
            song_interval.save_feature_stats(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/", file_name="Stats_" + str(AcousticsDirectory.split("/")[-1][:-4]), features=FeatureList)
        AcousticsProgress.config(text="Acoustic Calculations Complete!")
    if AcousticsInputVar.get() == 1: # Process whole folder
        AllSongFiles = glob.glob(str(AcousticsFileDisplay.cget("text")) + "/*.wav", recursive=True)
        try:
            os.makedirs(AcousticsOutput_Text.cget("text")+"/Acoustics/")
        except:
            pass
        Acoustic_LB = IntVar()
        AcousticsLoadingbar = ttk.Progressbar(AcousticsMainFrameSingle, mode="determinate", maximum=len(AllSongFiles), variable=Acoustic_LB)
        AcousticsLoadingbar.grid(row=19, column=1, columnspan=2)
        for file in AllSongFiles:
            AcousticsLoadingbar.step(1)
            AcousticsLoadingbar.update()
            time.sleep(0.1)
            AcousticsLoadingbar.update()
            AcousticsProgress.config(text="Processing file "+str(int(AllSongFiles.index(file))+1) + " of " +str(len(AllSongFiles)))
            song = dataloading.SongFile(file)
            audioread.audio_open(file).duration
            song_interval = acoustics.SongInterval(song, onset=0, offset=None,
                                                   win_length=int(win_length_entry_Acoustics.get()),
                                                   hop_length=int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()),
                                                   max_F0=int(max_F0_entry_Acoustics.get()),
                                                   min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=float(freq_range_entry_Acoustics.get()),
                                                   baseline_amp=int(baseline_amp_entry_Acoustics.get()),
                                                   fmax_yin=int(fmax_yin_entry_Acoustics.get()))
            if ContCalc.get() == 0:
                song_interval.save_features(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/Acoustics/",file_name=str((file.split("/")[-1][:-4]).split("\\")[-1]), features=FeatureList)
            if ContCalc.get() == 1:
                song_interval.save_feature_stats(out_file_path=str(AcousticsOutput_Text.cget("text"))+"/Acoustics/",file_name=str((file.split("/")[-1][:-4]).split("\\")[-1]), features=FeatureList)
        AcousticsProgress.config(text="Saving...")

        if ContCalc.get() == 1: # Mean Calculations
            OutputFiles = glob.glob(str(AcousticsOutput_Text.cget("text"))+"/Acoustics/" + "*feature_stats.csv", recursive=False)
            AverageAcousticOutput = pd.DataFrame(columns=["File Name", "Directory"])
            i = 1
            StatsList = ("_mean", "_std", "_min", "_25%", "_50%", "_75%", "_max")
            for feature in FeatureList:
                for x in StatsList:
                    i += 1
                    AverageAcousticOutput.insert(i, str(feature+x),[])
            Acoustic_LB.set(0)
            AcousticsLoadingbar.config(maximum=len(OutputFiles))
            AcousticsLoadingbar.update()
            for file in OutputFiles:
                AcousticsLoadingbar.step(1)
                df = pd.read_csv(file)
                temp = file.replace("\\","/").split("/")
                directory = ""
                for i in temp[:-1]:
                    directory = directory+i+"/"
                NewRow = [file.replace("\\","/").split("/")[-1], directory]
                for feature in FeatureList:
                    for x in range(len(StatsList)):
                        NewRow.append(str(df[feature][x]))
                AverageAcousticOutput.loc[len(AverageAcousticOutput)] = NewRow
                os.remove(file)
            pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
            if pattern.search(AllSongFiles[0]) != None:
                Bird_ID = pattern.search(AllSongFiles[0]).group()
            AverageAcousticOutput.to_csv(str(AcousticsOutput_Text.cget("text"))+"/Acoustics/"+Bird_ID+"_MeanValues_AllFiles.csv")
            MetaDataFiles = glob.glob(str(AcousticsOutput_Text.cget("text"))+"/Acoustics/" + "*metadata.csv", recursive=False)
            MasterMetaData = pd.read_csv(MetaDataFiles[0])
            os.remove(MetaDataFiles[0])
            for file in MetaDataFiles[1:]:
                # NewMetaRow = []
                # temp_df = pd.read_csv(file)
                # for column in MasterMetaData.columns:
                #     NewMetaRow.append(temp_df[column][0])
                # MasterMetaData.loc[len(MasterMetaData)] = NewMetaRow
                os.remove(file)
            try:
                MasterMetaData = MasterMetaData.drop("Date", axis=1)
            except:
                pass
            try:
                MasterMetaData = MasterMetaData.drop("file", axis=1)
            except:
                pass
            try:
                MasterMetaData = MasterMetaData.drop("onset", axis=1)
            except:
                pass
            try:
                MasterMetaData = MasterMetaData.drop("offset", axis=1)
            except:
                pass
            for column in MasterMetaData.columns:
                if "Unnamed" in column:
                    MasterMetaData = MasterMetaData.drop(column, axis=1)
            MasterMetaData.to_csv(str(AcousticsOutput_Text.cget("text")) + "/Acoustics/" + Bird_ID+"_Metadata_AllFiles.csv")
        if ContCalc.get() == 0: # Continuous calculations
            pass # Doesn't make sense to take large dataframes and combine them into one massive file -- it's better to keep them separate
    try:
        AcousticsLoadingbar.destroy()
    except:
        pass
    AcousticsProgress.config(text="Acoustic Calculations Complete!")
    # time.sleep(3)
    # AcousticsProgress.destroy()

def Acoustics_Syllables(RunAll=False):
    if RunAll==False:
        global MultiAcousticsDirectory
        global MultiAcousticsBirdID
        global Bird_ID
        global MultiAcousticsSongPath
        try:
            AcousticsLoadingBar.destroy()
        except: pass
        try:
            AcousticsLoadingMessage.destroy()
        except: pass

        Bird_ID = MultiAcousticsBirdID.get()
        global MultiAcousticsInputDir
        MultiAcousticsDirectory = MultiAcousticsInputDir.cget("text")
        if ".csv" not in MultiAcousticsDirectory:
            AcousticsLoadingMessage.config(text="Invalid Input File")
            AcousticsLoadingMessage.update()
        else:
            # Check that at least one feature is selected
            if (MultiRunGoodness.get()+MultiRunMean_frequency.get()+MultiRunEntropy.get()+MultiRunAmplitude.get()+
                MultiRunAmplitude_modulation.get()+MultiRunFrequency_modulation.get()+MultiRunPitch.get()) == 0:
                    AcousticsLoadingMessage.config(text="No Features Selected")
                    AcousticsLoadingMessage.update()
            else:
                # MultiAcousticsProgress = tk.Label(AcousticsMainFrameMulti, text="Calculating Acoustics...")
                # MultiAcousticsProgress.grid(row=25, column=1)
                # MultiAcousticsProgress.update()
                # time.sleep(1)
                try:
                    AcousticsLoadingMessage.destroy()
                except: pass
                try:
                    AcousticsLoadingBar.destroy()
                except: pass
                syll_df = pd.read_csv(str(MultiAcousticsDirectory))
                syll_df=syll_df.rename(columns={"onset":"onsets","offset":"offsets", "file":"files"})
                AcousticsLoadingMessage = tk.Label(AcousticsMainFrameMulti, text="Loading Data...")
                AcousticsLoadingMessage.grid(row=30, column=1)
                gui.update_idletasks()
                time.sleep(1)
                AcousticsLoadingBar = ttk.Progressbar(AcousticsMainFrameMulti, mode="determinate", maximum=len(syll_df["directory"].unique())+2)
                AcousticsLoadingBar.grid(row=31, column=1)
                AcousticsLoadingBar.update()
                gui.update_idletasks()

                global MultiAcousticsOutputDisplay
                MultiAcousticsOutputDirectory_temp = MultiAcousticsOutputDisplay.cget("text")
                try:
                    os.makedirs(MultiAcousticsOutputDirectory_temp+"/Acoustics_Syllables/")
                    MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory_temp+"/Acoustics_Syllables/"
                except:
                    MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory_temp+"/Acoustics_Syllables/"

                AcousticsLoadingBar.step(1)
                AcousticsLoadingBar.update()
                gui.update_idletasks()


                # MultiAcousticsOutputDirectory = MultiAcousticsOutputDirectory + "/"

                MultiAcousticsDirectory2 = ""
                for a in MultiAcousticsDirectory.split("/")[:-1]:
                    MultiAcousticsDirectory2 = MultiAcousticsDirectory2+a+"/"
                MultiAcousticsDirectory = MultiAcousticsDirectory2

                AcousticOutput = MultiAcousticsOutputDirectory

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

                # Try below function, assuming BirdID is chosen, otherwise grab bird ID from directory and proceed
                # song_paths_list = syll_df["directory"].unique()
                for column in syll_df.columns:
                    if "Unnamed" in column:
                        syll_df = syll_df.drop(column, axis=1)
                AcousticsLoadingMessage.config(text="Calculating Acoustic Features...")
                AcousticsLoadingMessage.update()

                AcousticsLoadingBar.step(1)
                AcousticsLoadingBar.update()
                gui.update_idletasks()
                acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df, song_folder_path=str(MultiAcousticsSongPath.cget("text")+"/"),
                                                       win_length=int(win_length_entry_Acoustics.get()),hop_length= int(hop_length_entry_Acoustics.get()),
                                                       n_fft=int(n_fft_entry_Acoustics.get()), max_F0=int(max_F0_entry_Acoustics.get()),min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                       freq_range=float(freq_range_entry_Acoustics.get()),baseline_amp=int(baseline_amp_entry_Acoustics.get()),fmax_yin=int(fmax_yin_entry_Acoustics.get()))
                acoustic_data.save_features(out_file_path=AcousticOutput,
                                            file_name=str(Bird_ID) + "_feature_table",
                                            features=FeatureList)
                acoustic_data.save_feature_stats(out_file_path=AcousticOutput,
                                                 file_name=str(Bird_ID) + "_syll_table",
                                                 features=FeatureList)
                # acoustic_data.save_features(out_file_path=AcousticOutput,
                #                             file_name=str(Bird_ID) + "_feature_table",
                #                             features=FeatureList)
                # acoustic_data.save_feature_stats(out_file_path=AcousticOutput,
                #                                  file_name=str(Bird_ID) + "_syll_table",
                #                                  features=FeatureList)
                # Generate Metadata File for Advanced Settings #
                AcousticsLoadingMessage.config(text="Saving Data...")
                Acoustic_SettingsNames = ["win_length", "hop_length",
                                          "n_fft_entry", "max_F0",
                                          "min_frequency",
                                          "freq_range", "baseline_amp",
                                          "fmax_yin"]
                Acoustic_SettingsValues = [win_length_entry_Acoustics.get(),
                                            hop_length_entry_Acoustics.get(),
                                            n_fft_entry_Acoustics.get(),
                                            max_F0_entry_Acoustics.get(),
                                            min_frequency_entry_Acoustics.get(),
                                            freq_range_entry_Acoustics.get(),
                                            baseline_amp_entry_Acoustics.get(),
                                            fmax_yin_entry_Acoustics.get()]
                # Acoustic_SettingsValues_df = pd.DataFrame({'Value': Acoustic_SettingsValues})

                setting_index = -1
                ValuesRow = []
                TitleRow = []
                for setting in Acoustic_SettingsNames:
                    setting_index+=1
                    TitleRow.append(setting)
                    ValuesRow.append(Acoustic_SettingsValues[setting_index])

                AcousticSettingsMetadata = pd.DataFrame({"win_length":[Acoustic_SettingsValues[0]],
                                                         "hop_length":[Acoustic_SettingsValues[1]],
                                                         "n_fft_entry":[Acoustic_SettingsValues[2]],
                                                         "max_F0":[Acoustic_SettingsValues[3]],
                                                         "min_frequency":[Acoustic_SettingsValues[4]],
                                                         "freq_range":[Acoustic_SettingsValues[5]],
                                                         "baseline_amp":[Acoustic_SettingsValues[6]],
                                                         "fmax_yin":[Acoustic_SettingsValues[7]]})
                AcousticsLoadingBar.step(1)
                AcousticsLoadingBar.update()
                gui.update_idletasks()
                # AcousticSettingsMetadata = pd.DataFrame(ValuesRow, columns=TitleRow)


                # AcousticSettingsMetadata = pd.concat([AcousticSettingsMetadata, Acoustic_SettingsValues], axis=1)
                # AcousticSettingsMetadata.to_csv(
                #     AcousticOutput + str(Bird_ID) + "_AcousticSettings_Metadata.csv")

                if RunAll == False:
                    AcousticsLoadingMessage.config(text="Acoustics Calculations Complete!")
                    AcousticsLoadingBar.destroy()
                    gui.update_idletasks()
                    # time.sleep(5)
                    # AcousticsLoadingMessage.destroy()
    if RunAll==True:
        global RunAll_InputFileDisplay
        global RunAll_OutputDir
        global All_Bird_ID
        global RunAll_SongPath

        Bird_ID = All_Bird_ID
        FeatureList = ["Goodness","Mean_frequency","Entropy","Amplitude","Amplitude_modulation","Frequency_modulation","Pitch"]
        syll_df = pd.read_csv(str(RunAll_InputFileDisplay.cget("text")))
        syll_df = syll_df.rename(columns={"onset": "onsets", "offset": "offsets", "file": "files"})

        # Try below function, assuming BirdID is chosen, otherwise grab bird ID from directory and proceed
        try:
            acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df,
                                                   song_folder_path=str(RunAll_SongPath.cget("text"))+"/",
                                                   win_length=int(win_length_entry_Acoustics.get()),
                                                   hop_length=int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()),
                                                   max_F0=int(max_F0_entry_Acoustics.get()),
                                                   min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=float(freq_range_entry_Acoustics.get()),
                                                   baseline_amp=int(baseline_amp_entry_Acoustics.get()),
                                                   fmax_yin=int(fmax_yin_entry_Acoustics.get()))
        except:
            pattern = re.compile('(?i)[A-Z][0-9][0-9][0-9]')
            Bird_ID = str(pattern.search(RunAll_InputFileDisplay.cget("text")).group())
            acoustic_data = acoustics.AcousticData(Bird_ID=str(Bird_ID), syll_df=syll_df,
                                                   song_folder_path=str(RunAll_SongPath.cget("text"))+"/",
                                                   win_length=int(win_length_entry_Acoustics.get()),
                                                   hop_length=int(hop_length_entry_Acoustics.get()),
                                                   n_fft=int(n_fft_entry_Acoustics.get()),
                                                   max_F0=int(max_F0_entry_Acoustics.get()),
                                                   min_frequency=int(min_frequency_entry_Acoustics.get()),
                                                   freq_range=float(freq_range_entry_Acoustics.get()),
                                                   baseline_amp=int(baseline_amp_entry_Acoustics.get()),
                                                   fmax_yin=int(fmax_yin_entry_Acoustics.get()))
        AcousticOutput = RunAll_OutputDir+"/Acoustics/"
        try:
            os.makedirs(AcousticOutput)
        except: pass
        acoustic_data.save_features(out_file_path=AcousticOutput,
                                    file_name=str(Bird_ID) + "_feature_table",
                                    features=FeatureList)
        acoustic_data.save_feature_stats(out_file_path=AcousticOutput, file_name=str(Bird_ID) + "_syll_table",
                                         features=FeatureList)
        # Generate Metadata File for Advanced Settings #
        AcousticSettingsMetadata = pd.DataFrame()
        Acoustic_SettingsNames = ["win_length_entry_Acoustics", "hop_length_entry_Acoustics",
                                  "n_fft_entry_Acoustics", "max_F0_entry_Acoustics",
                                  "min_frequency_entry_Acoustics",
                                  "freq_range_entry_Acoustics", "baseline_amp_entry_Acoustics",
                                  "fmax_yin_entry_Acoustics"]
        AcousticSettingsMetadata["Main"] = Acoustic_SettingsNames

        Acoustic_SettingsValues = pd.DataFrame({'Value': [win_length_entry_Acoustics.get(),
                                                          hop_length_entry_Acoustics.get(),
                                                          n_fft_entry_Acoustics.get(), max_F0_entry_Acoustics.get(),
                                                          min_frequency_entry_Acoustics.get(),
                                                          freq_range_entry_Acoustics.get(),
                                                          baseline_amp_entry_Acoustics.get(),
                                                          fmax_yin_entry_Acoustics.get()]})

        AcousticSettingsMetadata = pd.concat([AcousticSettingsMetadata, Acoustic_SettingsValues], axis=1)
        AcousticSettingsMetadata.to_csv(AcousticOutput + str(Bird_ID) + "_AcousticSettings_Metadata.csv")

def Syntax(RunAll=False, Mode="Syntax"):
    import avn.syntax as syntax
    import avn.plotting as plotting
    global syntax_data

    if RunAll == False:
        global SyntaxDirectory
        global DropCalls
        global SyntaxFileDisplay
        global SyntaxSongLocation
        try:
            SyntaxProgress.destroy()
        except: pass
        try:
            SyntaxProgressBar.destroy()
        except: pass
        SyntaxProgress = tk.Label(SyntaxMainFrame, text="Running...")
        SyntaxProgress.grid(row=10, column=0, columnspan=2)
        SyntaxProgressBar = ttk.Progressbar(SyntaxMainFrame, mode="determinate", maximum=5)
        SyntaxProgressBar.grid(row=11, column=0, columnspan=2)

        if Mode == "Syntax":
            syll_df = pd.read_csv(SyntaxFileDisplay.cget("text"))
            Bird_ID = SyntaxBirdID.get()
        if Mode == "Heatmap":
            global Syntax_HeatmapTab_Input
            syll_df = pd.read_csv(Syntax_HeatmapTab_Input.cget("text"))
            Bird_ID = Syntax_HeatmapTab_BirdID.get()
        if Mode == "Raster":
            global Syntax_Raster_Input
            syll_df = pd.read_csv(Syntax_Raster_Input.cget("text"))
            Bird_ID = Syntax_Raster_BirdID.get()

        merged_syntax_data = pd.DataFrame()
        # Syntax_DirIndex = 0
        # for directory in syll_df["directory"].unique():
        if Mode == "Syntax":
            directory = SyntaxSongLocation.cget("text")+"/"
            global SyntaxOutputDisplay
            SyntaxOutputFolder = SyntaxOutputDisplay.cget("text") + "/Syntax_" + SyntaxBirdID.get() + str(
                directory.split("/")[-1]) + "/"
            try:
                os.makedirs(SyntaxOutputFolder)
            except:
                pass
        if Mode == "Heatmap":
            directory = Syntax_HeatmapTab_SongLocation.cget("text") + "/"
        if Mode == "Raster":
            directory = Syntax_Raster_SongLocation.cget("text")+"/"


        SyntaxProgressBar.step(1)
        SyntaxProgressBar.update()
        # Make dataframe for current directory
        # syll_df_temp = syll_df[syll_df["directory"] == directory]
        syll_df_temp = syll_df
        for column in syll_df_temp.columns:
            if "Unnamed" in column:
                syll_df_temp = syll_df_temp.drop(column, axis=1)
        Bird_ID_Blank = ""
        syntax_data = syntax.SyntaxData(Bird_ID_Blank, syll_df_temp)
        dir_corrected = directory.replace("\\", "/")[:-1]
        syntax_data.add_file_bounds(dir_corrected)
        syntax_data.add_gaps(min_gap=float(min_gap_entry_Syntax.get()))
        gaps_df = syntax_data.get_gaps_df()
        SyntaxProgressBar.step(1)
        SyntaxProgressBar.update()
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
        SyntaxProgressBar.step(1)
        SyntaxProgressBar.update()


        # syntax_analysis_metadata = syntax_data.save_syntax_data(tempSyntaxDirectory2)
        #syntax_data["directory"] = np.array(len(syntax_data["directory"]*directory))

        # Prev_df = pd.read_csv(SyntaxOutputFolder+"/temp.csv")
        # merged_syntax_data.save_syntax_data(SyntaxOutputFolder)
        # Curr_df = pd.read_csv(SyntaxOutputFolder+"")
        if Mode == "Syntax":
            FileSuffixes = ["_per_syll_stats.csv","_syll_df.csv","_syntax_analysis_metadata.csv",
                            "_syntax_entropy.csv","_trans_mat.csv","_trans_mat_prob.csv"]
            Files = glob.glob(SyntaxOutputFolder + "*.csv")
            DuplicateFiles = False
            for file in Files:
                if Bird_ID+file in Files:
                    DuplicateFiles = True
            if DuplicateFiles == False:
                syntax_data.save_syntax_data(SyntaxOutputFolder)

                SyntaxProgressBar.step(1)
                SyntaxProgressBar.update()
                # merged_syll_df.to_csv(SyntaxOutputFolder+str(SyntaxBirdID)+"_syll_df.csv")
                entropy_df = pd.DataFrame({'Entropy Rate':[entropy_rate], 'Entropy Rate Normalized':[entropy_rate_norm]})
                entropy_df.to_csv(SyntaxOutputFolder+str(SyntaxBirdID.get())+"_syntax_entropy.csv")
                per_syll_stats.to_csv(SyntaxOutputFolder+str(SyntaxBirdID.get())+"_per_syll_stats.csv")
                # merged_syntax_analysis_metadata.to_csv(SyntaxOutputFolder+str(SyntaxBirdID)+"_syntax_analysis_metadata.csv")
                # merged_trans_mat.to_csv(SyntaxOutputFolder+str(SyntaxBirdID)+"_trans_mat.csv")
                # merged_trans_mat_prob.to_csv(SyntaxOutputFolder+str(SyntaxBirdID)+"_trans_mat_prob.csv")
                # except:
                # syntax_data.save_syntax_data(SyntaxOutputFolder)
                SyntaxProgressBar.step(1)
                SyntaxProgressBar.update()
                Bird_ID = str(SyntaxBirdID.get())
                for file in glob.glob(SyntaxOutputFolder + "*.csv"):
                    if Bird_ID not in file.replace("\\","/").split("/")[-1]:
                        NewFileName = ""
                        for i in file.replace("\\","/").split("/")[:-1]:
                            NewFileName = NewFileName+i+"/"
                        NewFileName = NewFileName+Bird_ID+file.replace("\\", "/").split("/")[-1]
                        os.rename(file, NewFileName)
            else:
                SyntaxProgress.config(text="Warning! Duplicate output files detected in output directory.\n "
                                           "Please delete files or move to alternate folder.")
                SyntaxProgressBar.destroy()

        if Mode != "Syntax":
            SyntaxProgress.config(text="Complete!")
            time.sleep(0.1)
            SyntaxProgressBar.destroy()
            time.sleep(3)
            # SyntaxProgress.destroy()
            gui.update_idletasks()
        elif DuplicateFiles == False:
            SyntaxProgress.config(text="Complete!")
            time.sleep(0.1)
            SyntaxProgressBar.destroy()
            time.sleep(3)
            # SyntaxProgress.destroy()
            gui.update_idletasks()

    if RunAll == True:
        global RunAll_InputFileDisplay
        global All_Bird_ID
        global RunAll_SongPath

        Bird_ID = All_Bird_ID
        syll_df = pd.read_csv(RunAll_InputFileDisplay.cget("text"))

        merged_syntax_data = pd.DataFrame()
        # Syntax_DirIndex = 0
        # for directory in syll_df["directory"].unique():
        #     Syntax_DirIndex += 1
        directory = str(RunAll_SongPath.cget("text"))+"/"
        global RunAll_OutputDir
        try:
            SyntaxOutputFolder = RunAll_OutputDir + "Syntax/"
            os.makedirs(SyntaxOutputFolder)
        except:
            pass

        # Make dataframe for current directory
        # syll_df_temp = syll_df[syll_df["directory"] == directory]
        syll_df_temp = syll_df

        try:
            syll_df_temp.rename(columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
        except: pass
        syntax_data = syntax.SyntaxData("", syll_df_temp)
        dir_corrected = directory.replace("\\", "/")[:-1]
        syntax_data.add_file_bounds(dir_corrected)
        syntax_data.add_gaps(min_gap=float(min_gap_entry_Syntax.get()))
        gaps_df = syntax_data.get_gaps_df()
        syntax_data.drop_calls()
        syntax_data.make_transition_matrix()
        entropy_rate = syntax_data.get_entropy_rate()
        entropy_rate_norm = entropy_rate / np.log2(len(syntax_data.unique_labels) + 1)
        prob_repetitions = syntax_data.get_prob_repetitions()
        single_rep_counts, single_rep_stats = syntax_data.get_single_repetition_stats()
        intro_notes_df = syntax_data.get_intro_notes_df()
        prop_sylls_in_short_bouts = syntax_data.get_prop_sylls_in_short_bouts(max_short_bout_len=2)
        per_syll_stats = syntax.Utils.merge_per_syll_stats(single_rep_stats, prop_sylls_in_short_bouts,
                                                           intro_notes_df)
        pair_rep_counts, pair_rep_stats = syntax_data.get_pair_repetition_stats()
        tempSyntaxDirectory2 = ""

        entropy_df = pd.DataFrame({'Entropy Rate': [entropy_rate], 'Entropy Rate Normalized': [entropy_rate_norm]})
        entropy_df.to_csv(SyntaxOutputFolder + Bird_ID + "_syntax_entropy.csv")
        per_syll_stats.to_csv(SyntaxOutputFolder + Bird_ID + "_per_syll_stats.csv")

        # syntax_analysis_metadata = syntax_data.save_syntax_data(tempSyntaxDirectory2)
        # syntax_data["directory"] = np.array(len(syntax_data["directory"]*directory))

        # Prev_df = pd.read_csv(SyntaxOutputFolder+"/temp.csv")
        # merged_syntax_data.save_syntax_data(SyntaxOutputFolder)
        # Curr_df = pd.read_csv(SyntaxOutputFolder+"")
        # syntax_data.save_syntax_data(SyntaxOutputFolder)
        # temp_file_list = []
        # all_files_list = glob.glob(SyntaxOutputFolder + "/*.csv")

        # merged_syll_df.to_csv(SyntaxOutputFolder + Bird_ID + "_syll_df.csv")
        # merged_syntax_analysis_metadata.to_csv(SyntaxOutputFolder + Bird_ID + "_syntax_analysis_metadata.csv")
        # merged_trans_mat.to_csv(SyntaxOutputFolder + Bird_ID + "_trans_mat.csv")
        # merged_trans_mat_prob.to_csv(SyntaxOutputFolder + Bird_ID + "_trans_mat_prob.csv")
        # syntax_data.save_syntax_data(SyntaxOutputFolder)

        DuplicateFiles = False
        Files = glob.glob(SyntaxOutputFolder + "*.csv")
        for file in Files:
            if Bird_ID + file in Files:
                DuplicateFiles = True
        if DuplicateFiles == False:
            syntax_data.save_syntax_data(SyntaxOutputFolder)

        # for file in glob.glob(SyntaxOutputFolder + "*.csv"):
        #     if Bird_ID not in file.replace("\\", "/").split("/")[-1]:
        #         NewFileName = ""
        #         for i in file.replace("\\", "/").split("/")[:-1]:
        #             NewFileName = NewFileName + i + "/"
        #         NewFileName = NewFileName + Bird_ID + file.replace("\\", "/").split("/")[-1]
        #         os.rename(file, NewFileName)


    return syntax_data

def PrintPlainSpectrograms():
    global PlainDirectory
    global PlainDirectoryLabel
    PlainDirectory = PlainOutputFolder_Label.cget('text')+ "/Unlabeled_Spectrograms_WholeFolder/"
    try:
        os.makedirs(PlainDirectory)
    except: pass
    FileList = glob.glob(str(PlainDirectoryLabel.cget('text')) + "/*.wav")
    try:
        PlainProgressLabel.destroy()
    except: pass
    try:
        PlainProgressBar.destroy()
    except: pass
    PlainProgressLabel = tk.Label(Plain_Folder, text="Generating Spectrograms...")
    PlainProgressLabel.grid(row=10, column=0, columnspan=2)
    PlainProgressBar = ttk.Progressbar(Plain_Folder, mode="determinate", maximum=len(FileList))
    PlainProgressBar.grid(row=11, column=0, columnspan=2)

    def make_spec(syll_wav, hop_length, win_length, n_fft, amin, ref_db, min_level_db):
        spectrogram = librosa.stft(syll_wav, hop_length=hop_length, win_length=win_length,
                                   n_fft=n_fft)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), amin=amin, ref=ref_db)

        # normalize
        S_norm = np.clip((spectrogram_db - min_level_db) / -min_level_db, 0, 1)

        return S_norm

    f = 0
    # compute spectrogram for each syllable
    syllables_spec = []

    for file in FileList:
        f+=1
        PlainProgressLabel.config(text="Generating Spectrograms ("+str(f)+" of "+str(len(FileList))+")")
        PlainProgressLabel.update()
        time.sleep(0.1)
        song = avn.dataloading.SongFile(file)
        # syllable_spec = make_spec(song,
        #                           hop_length=int(hop_length_entry_Plain.get()),
        #                           win_length=int(win_length_entry_Plain.get()),
        #                           n_fft=int(n_fft_entry_Plain.get()),
        #                           ref_db=int(ref_db_entry_Plain.get()),
        #                           amin=float(a_min_entry_Plain.get()),
        #                           min_level_db=int(min_level_db_entry_Plain.get()))
        #
        # # syllables_spec.append(syllable_spec)
        #
        spectrogram_db = avn.plotting.make_spectrogram(song)
        plt = avn.plotting.plot_spectrogram(spectrogram_db, song.sample_rate) # Modified avn.plotting.plot_spectrograms to return plot

        plt.savefig(PlainDirectory+str(file.split("/")[-1].split("\\")[-1])[:-4]+".png")
        plt.clf()
        PlainProgressBar.step(1)
        PlainProgressBar.update()

        # # normalize spectrograms
        # def norm(x):
        #     return (x - np.min(x)) / (np.max(x) - np.min(x))
        #
        # syllables_spec_norm = [norm(syllable_spec)]
        #
        # # Pad spectrograms for uniform dimensions
        # spec_lens = [np.shape(i)[1] for i in syllables_spec]
        # pad_length = np.max(spec_lens)
        #
        # syllables_spec_padded = []
        #
        # for spec in syllables_spec_norm:
        #     to_add = pad_length - np.shape(spec)[1]
        #     pad_left = np.floor(float(to_add) / 2).astype("int")
        #     pad_right = np.ceil(float(to_add) / 2).astype("int")
        #     spec_padded = np.pad(spec, [(0, 0), (pad_left, pad_right)], 'constant', constant_values=0)
        #     syllables_spec_padded.append(spec_padded)
        #
        # # flatten the spectrograms into 1D
        # specs_flattened = [spec.flatten() for spec in syllables_spec_padded]
        # specs_flattened_array = np.array(specs_flattened)
        #
        #

    PlainProgressLabel.config(text="Spectrograms Saved!")
    PlainProgressBar.destroy()
    gui.update_idletasks()

def PrintPlainSpectrogramsAlt():
    global PlainDirectoryAlt
    SaveLocation = ""
    for x in PlainDirectoryAlt[0].split("/")[:-1]:
        SaveLocation = SaveLocation+x+"/"
    PlainDirectory = PlainOutputAlt_Label.cget("text") + "/Unlabeled_Spectrograms_IndividualFiles/"
    try:
        os.makedirs(PlainDirectory)
    except: pass
    # FileList = glob.glob(str(PlainDirectoryLabel.cget('text')) + "/*.wav")
    try:
        PlainProgressLabel.destroy()
    except: pass
    try:
        PlainProgressBar.destroy()
    except: pass
    PlainProgressLabel = tk.Label(PlainSpectroAlt, text="Generating Spectrograms...")
    PlainProgressLabel.grid(row=10, column=0, columnspan=2)
    PlainProgressBar = ttk.Progressbar(PlainSpectroAlt, mode="determinate", max=len(PlainDirectoryAlt))
    PlainProgressBar.grid(row=11, column=0, columnspan=2)
    gui.update_idletasks()

    f = 0
    for file in PlainDirectoryAlt:
        f+=1
        PlainProgressLabel.config(text="Generating Spectrograms ("+str(f)+" of "+str(len(PlainDirectoryAlt))+")")
        PlainProgressLabel.update()
        time.sleep(0.1)
        song = avn.dataloading.SongFile(file)
        spectrogram_db = avn.plotting.make_spectrogram(song)
        plt = avn.plotting.plot_spectrogram(spectrogram_db, song.sample_rate) # Modified avn.plotting.plot_spectrograms to return plot

        plt.savefig(PlainDirectory+str(file.split("/")[-1].split("\\")[-1])[:-4]+".png")
        plt.clf()
        PlainProgressBar.step(1)
        PlainProgressBar.update()
    PlainProgressLabel.config(text="Spectrograms Saved!")
    PlainProgressBar.destroy()
    gui.update_idletasks()

def Timing(RunAll=False):
    BadInput = False
    if RunAll == False:
        try:
            TimingLoadingMessage.destroy()
        except: pass
        try:
            TimingLoadingBar.destroy()
        except: pass
        global TimingSongPath
        if ".csv" not in TimingInput_Text.cget("text"):
            BadInput = True
            print("Invalid Segmentation File!")
        global Timing_BirdID
        if (Timing_BirdID.get() == None or Timing_BirdID.get() == "Bird ID"):
            BadInput = True
            print("Invalid Bird ID!")
        if BadInput == False:
            TimingLoadingMessage = tk.Label(SyllableTiming, text="Calculating...")
            TimingLoadingMessage.grid(row=10, columnspan=2)
            TimingLoadingMessage.update()
            TimingLoadingBar = ttk.Progressbar(SyllableTiming, mode="indeterminate")
            TimingLoadingBar.grid(row=11, columnspan=2)
            TimingLoadingBar.update()
            TimingLoadingBar.start()
            gui.update_idletasks()
            time.sleep(1)
            syll_df = pd.read_csv(TimingInput_Text.cget("text"))
            Bird_ID = ""
            syll_df = syll_df.rename(
                columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
            try:
                os.makedirs(TimingOutput_Text.cget("text")+"/Timing/")
            except:
                pass
            Temp_TimingOutputFolder = TimingOutput_Text.cget("text")+"/Timing/"
            # DirectoryCount = 0
            gui.update_idletasks()

            Timing_df = pd.DataFrame({"directory":[],"syll_duration_entropy":[],"gap_duration_entropy":[]})

            # for directory in syll_df["directory"].unique():
            directory = TimingSongPath.cget("text")+"/"
            # DirectoryCount +=1
            Temp_Array = []
            Temp_Array.append(directory)
            TimingOutputFolder = Temp_TimingOutputFolder
            try:
                os.makedirs(TimingOutputFolder)
            except:
                pass
            # syll_df_temp = syll_df[syll_df["directory"]==directory]
            syll_df_temp = syll_df
            segment_timing = avn.timing.SegmentTiming(Bird_ID, syll_df_temp,
                                                      song_folder_path=directory)
            syll_durations = segment_timing.get_syll_durations()
            Bird_ID = Timing_BirdID.get()
            syll_durations.to_csv(TimingOutputFolder + Bird_ID + "_SyllDurations.csv")
            # There seems to be a conflict b/w seaborn (sns) and pandas, so the sns plotting functions raise an error... Everything else works fine
            sns.kdeplot(data=syll_durations, x='durations', bw_adjust=0.1)
            plt.title('Syllable Durations')
            plt.xlabel('Syllable duration (s)');
            plt.savefig(TimingOutputFolder + Bird_ID + "_SyllableDurations.png")
            plt.clf()
            syll_duration_entropy = segment_timing.calc_syll_duration_entropy()
            Temp_Array.append(syll_duration_entropy)
            gap_durations = segment_timing.get_gap_durations(max_gap=0.2)
            gap_durations.to_csv(TimingOutputFolder + Bird_ID + "_GapDurations.csv")
            sns.kdeplot(data=gap_durations, x='durations', bw_adjust=0.1)
            gui.update_idletasks()

            plt.title('Gap Durations')
            plt.xlabel('Gap duration (s)');
            plt.savefig(TimingOutputFolder + Bird_ID + "_GapDurations.png")
            plt.clf()

            gap_duration_entropy = segment_timing.calc_gap_duration_entropy()
            Temp_Array.append(gap_duration_entropy)

            TimingMetadata = pd.DataFrame({"Date": [str(date.today().strftime("%m/%d/%y"))],
                                           "avn_version": ["0.5.1"],
                                           "gui_version": ["0.1.0"],
                                           "max_gap": [str(max_gap_Entry.get())]
            })
            TimingMetadata.to_csv(TimingOutputFolder + Bird_ID + "_Timing_Metadata.csv")
            TimingLoadingMessage.config(text="Calculations Complete!")
            TimingLoadingBar.destroy()
            time.sleep(3)
            # TimingLoadingMessage.destroy()
    elif RunAll == True:
        global RunAll_InputFileDisplay
        global OutputDir
        global RunAllOut
        global RunAll_OutputDir
        global RunAll_SongPath
        syll_df = pd.read_csv(RunAll_InputFileDisplay.cget("text"))
        global All_Bird_ID
        Bird_ID = All_Bird_ID
        syll_df = syll_df.rename(
            columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files"})
        try:
            os.makedirs(TimingOutput_Text.cget("text") + "/Timing/")
        except:
            pass
        Temp_TimingOutputFolder = RunAll_OutputDir + "Timing/"
        try:
            os.makedirs(Temp_TimingOutputFolder)
        except:
            pass
        # DirectoryCount = 0

        Timing_df = pd.DataFrame({"directory": [], "syll_duration_entropy": [], "gap_duration_entropy": []})

        # for directory in syll_df["directory"].unique():
        #     DirectoryCount += 1
        directory = RunAll_SongPath.cget('text')+"/"
        # Temp_Array = []
        # Temp_Array.append(directory)
        TimingOutputFolder = Temp_TimingOutputFolder
        try:
            os.makedirs(TimingOutputFolder)
        except:
            pass
        # syll_df_temp = syll_df[syll_df["directory"] == directory]
        syll_df_temp = syll_df
        segment_timing = avn.timing.SegmentTiming("", syll_df_temp,
                                                  song_folder_path=directory)
        syll_durations = segment_timing.get_syll_durations()
        syll_durations.to_csv(TimingOutputFolder + Bird_ID+"_syll_durations.csv")
        # There seems to be a conflict b/w seaborn (sns) and pandas, so the sns plotting functions raise an error... Everything else works fine
        sns.kdeplot(data=syll_durations, x='durations', bw_adjust=0.1)
        plt.title('Syllable Durations')
        plt.xlabel('Syllable duration (s)');
        plt.savefig(TimingOutputFolder + str(Bird_ID)+"_Syllable_Durations.png")
        plt.clf()
        syll_duration_entropy = segment_timing.calc_syll_duration_entropy()
        # Temp_Array.append(syll_duration_entropy)
        gap_durations = segment_timing.get_gap_durations(max_gap=float(max_gap_Entry.get()))
        plt.savefig(TimingOutputFolder + str(Bird_ID)+"_Gap_Durations.png")
        sns.kdeplot(data=gap_durations, x='durations', bw_adjust=0.1)

        plt.title('Gap Durations')
        plt.xlabel('Gap duration (s)');
        plt.savefig(TimingOutputFolder + Bird_ID+"_GapDurations.png")

        gap_duration_entropy = segment_timing.calc_gap_duration_entropy()
        # Temp_Array.append(gap_duration_entropy)

        TimingMetadata = pd.DataFrame({"date": [str(date.today().strftime("%m/%d/%y"))],
                                       "avn_version": ["0.5.1"],
                                       "gui_version": ["0.1.0"],
                                       "max_gap": [str(max_gap_Entry.get())]
                                       })
        TimingMetadata.to_csv(TimingOutputFolder + Bird_ID + "_Timing_Metadata.csv")

        # # Create rhythm analysis object
        # rhythm_analysis = avn.timing.RhythmAnalysis(Bird_ID)
        # rhythm_spectrogram = rhythm_analysis.make_rhythm_spectrogram(
        #     song_folder_path=directory)
        #
        # fig_rhythm_spectrogram = rhythm_analysis.plot_rhythm_spectrogram()
        # fig_rhythm_spectrogram.savefig(TimingOutputFolder + Bird_ID+"_RhythmSpectrogram.png")
        # rhythm_spectrogram_entropy = rhythm_analysis.calc_rhythm_spectrogram_entropy()
        # peak_frequencies = rhythm_analysis.get_refined_peak_frequencies(freq_range=3)
        # fig_peak_frequencies = rhythm_analysis.plot_peak_frequencies()
        # fig_peak_frequencies.savefig(TimingOutputFolder + Bird_ID+"_PeakFrequencies.png")
        # peak_frequency_cv = rhythm_analysis.calc_peak_frequency_cv()
        # Timing_df.loc[len(Timing_df.index)] = Temp_Array
    if BadInput == True:
        pass
        # print("Invalid Timing Input!")

def TimingRhythm(RunAll=False):
    if RunAll == False:
        # Create rhythm analysis object
        Bird_ID = RhythmSpectrogram_BirdID.get()
        try:
            os.makedirs(RhythmSpectrogram_Output.cget("text") + "/Timing/")
        except:
            pass
        TimingOutputFolder = RhythmSpectrogram_Output.cget("text")+"/Timing/"
        rhythm_analysis = avn.timing.RhythmAnalysis(Bird_ID)
        rhythm_spectrogram = rhythm_analysis.make_rhythm_spectrogram(
            song_folder_path=str(RhythmSpectrogram_Input.cget("text")+"/"))

    elif RunAll == True:
        global RunAll_OutputDir
        TimingOutputFolder = RunAll_OutputDir + "Timing/"
        global All_Bird_ID
        Bird_ID = All_Bird_ID
        global RunAll_SongPath
        SongPath = RunAll_SongPath.cget("text")+"/"
        try:
            os.makedirs(TimingOutputFolder)
        except:
            pass

        rhythm_analysis = avn.timing.RhythmAnalysis(Bird_ID)
        rhythm_spectrogram = rhythm_analysis.make_rhythm_spectrogram(
            song_folder_path=str(SongPath))


    fig_rhythm_spectrogram = rhythm_analysis.plot_rhythm_spectrogram(smoothing_window=int(smoothing_window_Entry.get()))
    fig_rhythm_spectrogram.savefig(TimingOutputFolder + Bird_ID + "_RhythmSpectrogram.png")
    rhythm_spectrogram_entropy = rhythm_analysis.calc_rhythm_spectrogram_entropy()
    peak_frequencies = rhythm_analysis.get_refined_peak_frequencies(freq_range=3)
    fig_peak_frequencies = rhythm_analysis.plot_peak_frequencies()
    fig_peak_frequencies.savefig(TimingOutputFolder + Bird_ID + "_PeakFrequencies.png")
    peak_frequency_cv = rhythm_analysis.calc_peak_frequency_cv()

    TimingRhythm_df = pd.DataFrame({"rhythm_spectrogram_entropy":[rhythm_spectrogram_entropy],
                                    "peak_frequency_cv":[peak_frequency_cv]})
    TimingRhythm_df.to_csv(TimingOutputFolder + Bird_ID + "_Rhythm_features.csv")

    TimingRhythmMetadata = pd.DataFrame({"date": [str(date.today().strftime("%m/%d/%y"))],
                                   "avn_version": ["0.5.1"],
                                   "gui_version": ["0.1.0"],
                                   "smoothing_window": [str(smoothing_window_Entry.get())]
                                   })
    TimingRhythmMetadata.to_csv(TimingOutputFolder + Bird_ID + "_Timing_Rhythm_Metadata.csv")

def MoreInfo(Event):
    # print(Event.widget)
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
                    "14": {"3":"Select whether the GUI will return acoustic feature data as individual values across time bins or "
                               "an average across the entire trace"+"!Calculation Method"},
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
                    "21": {"2":"Minimum duration in seconds for a gap between syllables "
                                 "to be considered syntactically relevant. This value should "
                                 "be selected such that gaps between syllables in a bout are "
                                 "shorter than min_gap, but gaps between bouts are longer than min_gap"+"!min_gap"},
                    "25": {"2": "Lower cutoff frequency in Hz for a hamming window " \
                                "bandpass filter applied to the audio data before generating " \
                                "spectrograms. Frequencies below this value will be filtered out" + '!bandpass_lower_cutoff',
                           "4": "Upper cutoff frequency in Hz for a hamming window bandpass" \
                                " filter applied to the audio data before generating spectrograms. " \
                                "Frequencies above this value will be filtered out" + "!bandpass_upper_cutoff",
                           "6": "Minimum amplitude threshold in the spectrogram. Values " \
                                "lower than a_min will be set to a_min before conversion to decibels" + "!a_min",
                           "8": "When making the spectrogram and converting it from amplitude " \
                                "to db, the amplitude is scaled relative to this reference: " \
                                "20 * log10(S/ref_db) where S represents the spectrogram with amplitude values" + "!ref_db",
                           "10": "When making the spectrogram, once the amplitude has been converted " \
                                 "to decibels, the spectrogram is normalized according to this value: " \
                                 "(S - min_level_db)/-min_level_db where S represents the spectrogram " \
                                 "in db. Any values of the resulting operation which are <0 are set to " \
                                 "0 and any values that are >1 are set to 1" + "!min_level_db",
                           "12": "When making the spectrogram, this is the length of the windowed " \
                                 "signal after padding with zeros. The number of rows spectrogram is" \
                                 " \"(1+n_fft/2)\". The default value,\"n_fft=512\" samples, " \
                                 "corresponds to a physical duration of 93 milliseconds at a sample " \
                                 "rate of 22050 Hz, i.e. the default sample rate in librosa. This value " \
                                 "is well adapted for music signals. However, in speech processing, the " \
                                 "recommended value is 512, corresponding to 23 milliseconds at a sample" \
                                 " rate of 22050 Hz. In any case, we recommend setting \"n_fft\" to a " \
                                 "power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm" + "!n_fft",
                           "14": "When making the spectrogram, each frame of audio is windowed by a window " \
                                 "of length \"win_length\" and then padded with zeros to match \"n_fft\"." \
                                 " Padding is added on both the left- and the right-side of the window so" \
                                 " that the window is centered within the frame. Smaller values improve " \
                                 "the temporal resolution of the STFT (i.e. the ability to discriminate " \
                                 "impulses that are closely spaced in time) at the expense of frequency " \
                                 "resolution (i.e. the ability to discriminate pure tones that are closely" \
                                 " spaced in frequency). This effect is known as the time-frequency " \
                                 "localization trade-off and needs to be adjusted according to the " \
                                 "properties of the input signal" + "!win_length",
                           "16": "The number of audio samples between adjacent windows when creating " \
                                 "the spectrogram. Smaller values increase the number of columns in " \
                                 "the spectrogram without affecting the frequency resolution" + "!hop_length",
                           "18": "Maximum frequency in Hz used to estimate fundamental frequency " \
                                 "with the YIN algorithm" + "!max_spec_size"}
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

    ### Acoustic Features Calculation Method ###
    if Module == "14":
        AcousticsCalcMethod.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
        AcousticsCalcMethod.update()
        AcousticsCalcTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
        AcousticsCalcTitle.update()

    ### Acoustic Features ###
    if Module == "16":
        AcousticSettingsDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
        AcousticSettingsDialog.update()
        AcousticsSettingsDialogTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
        AcousticsSettingsDialogTitle.update()

    ### Syntax ###
    if Module == "21":
        SyntaxDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
        SyntaxDialog.update()
        SyntaxDialogTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
        SyntaxDialogTitle.update()

    ### Plain ###
    if Module == "25":
        PlainSettingsDialog.config(text=SettingsDict[Module][Setting].split("!")[0], wraplength=Text_wraplength)
        PlainSettingsDialog.update()
        PlainSettingsDialogTitle.config(text=SettingsDict[Module][Setting].split("!")[1])
        PlainSettingsDialogTitle.update()

def LessInfo(Event):
    if "frame10" in str(Event.widget):
        LabelingSpectrogramDialog.config(text="")
        LabelingSpectrogramTitle.config(text="")
    elif "frame11" in str(Event.widget):
        LabelingUMAPDialog.config(text="")
        LabelingUMAPTitle.config(text="")
    elif "frame12" in str(Event.widget):
        LabelingClusterDialog.config(text="")
        LabelingClusterTitle.config(text="")
    elif "frame14" in str(Event.widget):
        AcousticsCalcMethod.config(text="")
        AcousticsCalcTitle.config(text="")
    elif "frame16" in str(Event.widget):
        AcousticSettingsDialog.config(text="")
        AcousticsSettingsDialogTitle.config(text="")
    elif "frame 21" in str(Event.widget):
        SyntaxDialog.config(text="")
        SyntaxDialogTitle.config(text="")
    elif "frame 25" in str(Event.widget):
        PlainSettingsDialog.config(text="")
        PlainSettingsDialogTitle.config(text="")

def Validate_Settings(Widget, WidgetName, ErrorLabel):
    IntVarList = ['n_components_entry_Labeling', 'random_state_entry_Labeling', 'min_samples_entry_Labeling',
                  'n_components_entry_Plain', 'random_state_entry_Plain', 'min_samples_entry_Plain',
                  'win_length_entry_Acoustics',"hop_length_entry_Acoustics","n_fft_entry_Acoustics","max_F0_entry_Acoustics",
                  "min_frequency_entry_Acoustics","baseline_amp_entry_Acoustics","fmax_yin_entry_Acoustics"]
    FloatVarList = ['a_min_entry_Labeling', 'min_dist_entry_Labeling', 'min_gap_entry_Syntax', 'spread_entry_Labeling',
                    'a_min_entry_Plain', 'min_dist_entry_Plain', 'min_gap_entry_Syntax', 'spread_entry_Plain',
                    'freq_range_entry_Acoustics',"min_gap_entry_Syntax"]
    NormalList = ['bandpass_lower_cutoff_entry_Labeling', 'bandpass_upper_cutoff_entry_Labeling',
                  'ref_db_entry_Labeling', 'min_level_db_entry_Labeling','n_fft_entry_Labeling',
                  'win_length_entry_Labeling', 'hop_length_entry_Labeling',
                  'bandpass_lower_cutoff_entry_Plain', 'bandpass_upper_cutoff_entry_Plain',
                  'ref_db_entry_Plain', 'min_level_db_entry_Plain', 'n_fft_entry_Plain',
                  'win_length_entry_Plain', 'hop_length_entry_Plain']

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
    elif "min_cluster_prop" in WidgetName:
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
    elif "n_neighbors" in WidgetName:
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
    # else:
    #     try:
    #         int(Widget.get())
    #         ErrorLabel.config(text="")
    #         Widget.config(bg="white")
    #     except:
    #         ErrorLabel.config(text="Invalid Input: Please Enter An Integer")
    #         Widget.config(bg="red")

def SimilarityScoring_Prep():
    global Similarity_BirdID
    global SimilarityInput
    global SimilarityOutput
    global SimilaritySongPath
    try:
        SimilarityLoadingMessage.destroy()
    except:pass
    try:
        SimilarityLoadingBar.destroy()
    except:pass
    SimilarityLoadingMessage = tk.Label(PrepSpectrograms, text="Processing Syllables...")
    SimilarityLoadingMessage.grid(row=10, column=0, columnspan=2)
    SimilarityLoadingBar = ttk.Progressbar(PrepSpectrograms, mode="indeterminate")
    SimilarityLoadingBar.grid(row=11, column=0, columnspan=2)
    SimilarityLoadingBar.start()
    SimilarityLoadingBar.update()
    gui.update_idletasks()

    Bird_ID = Similarity_BirdID.get()
    segmentations = pd.read_csv(SimilarityInput.cget("text"))
    segmentations = segmentations.rename(columns={"onset": "onsets", "offset": "offsets", "cluster": "cluster", "file": "files","directory":"directory"})

    song_folder_path = SimilaritySongPath.cget("text")+"/"
    out_dir = str(SimilarityOutput.cget("text"))+"/"+Bird_ID+"_similarity_prep/"
    try:
        os.makedirs(out_dir)
    except:
        pass

    similarity.prep_spects(Bird_ID=Bird_ID, segmentations=segmentations, song_folder_path=song_folder_path,
                           out_dir=out_dir)
    SimilarityLoadingMessage.config(text="Syllable Spectrograms Saved!")
    SimilarityLoadingBar.destroy()
    time.sleep(3)
    # SimilarityLoadingMessage.destroy()
    gui.update_idletasks()

def SimilarityScoring_Output():
    global Similarity_BirdID2
    Bird_ID_2 = Similarity_BirdID2.get()
    global Similarity_BirdID3
    Bird_ID_3 = Similarity_BirdID3.get()
    global SimilarityInput2
    spectrograms_dir_2 = SimilarityInput2.cget("text")
    global SimilarityOutput2
    try:
        SimilarityProgressMessage.destroy()
    except: pass
    try:
        SimilarityProgressBar.destroy()
    except: pass
    SimilarityProgressMessage = tk.Label(OutputSimilarity, text="Calculating Similarity...")
    SimilarityProgressMessage.grid(row=10, column=2)
    SimilarityProgressBar = ttk.Progressbar(OutputSimilarity, mode="determinate", maximum=5)
    SimilarityProgressBar.grid(row=11, column=2)
    OutputFolder = str(SimilarityOutput2.cget("text"))+"/SimilarityComparison_"+Bird_ID_2+"-"+Bird_ID_3+"/"
    try:
        os.makedirs(OutputFolder)
    except: pass
    model = similarity.load_model()
    embeddings_2 = similarity.calc_embeddings(Bird_ID=Bird_ID_2,
                                            spectrograms_dir=spectrograms_dir_2,
                                            model=model)

    pca = PCA(n_components=2)
    embeddings_PCs_2 = pca.fit_transform(embeddings_2)
    plt.scatter(embeddings_PCs_2[:, 0], embeddings_PCs_2[:, 1], s=5,label = Bird_ID_2)
    plt.legend()
    plt.savefig(OutputFolder + Bird_ID_2+".png")
    plt.clf()
    SimilarityProgressBar.step()
    SimilarityProgressBar.update()
    ###

    global SimilarityInput3
    spectrograms_dir_3 = SimilarityInput3.cget("text")
    embeddings_3 = similarity.calc_embeddings(Bird_ID=Bird_ID_3,
                                            spectrograms_dir=spectrograms_dir_3,
                                            model=model)
    embeddings_PCs_3 = pca.fit_transform(embeddings_3)
    plt.scatter(embeddings_PCs_3[:, 0], embeddings_PCs_3[:, 1], s=5,label = Bird_ID_3)
    plt.legend()
    plt.savefig(OutputFolder+Bird_ID_3+".png")
    plt.clf()
    SimilarityProgressBar.step()
    SimilarityProgressBar.update()
    global SaveEmbedding
    if SaveEmbedding.get() == 1:
        embeddings_2_pd = pd.DataFrame(embeddings_2)
        embeddings_2_pd.to_csv(OutputFolder+Bird_ID_2+"_Embeddings.csv")
        embeddings_3_pd = pd.DataFrame(embeddings_3)
        embeddings_3_pd.to_csv(OutputFolder+Bird_ID_3+"_Embeddings.csv")

    ########
    SimilarityProgressBar.step()
    SimilarityProgressBar.update()
    pca.fit(np.concatenate((embeddings_2, embeddings_3)))
    embedding_PCs_2 = pca.transform(embeddings_2)
    embedding_PCs_3 = pca.transform(embeddings_3)
    plt.clf()
    plt.scatter(embedding_PCs_2[:, 0], embedding_PCs_2[:, 1], s=5, label=str(Bird_ID_2), alpha=0.5)
    plt.scatter(embedding_PCs_3[:, 0], embedding_PCs_3[:, 1], s=5, label=str(Bird_ID_3), alpha=0.5)
    plt.legend()
    plt.savefig(OutputFolder+"Combined.png")
    SimilarityProgressBar.step()
    SimilarityProgressBar.update()
    emd = similarity.calc_emd(embeddings_2, embeddings_3)
    emd_df = pd.DataFrame(data= {"EMD":[emd]})
    emd_df.to_csv(OutputFolder+"EMD.csv")
    SimilarityProgressBar.step()
    SimilarityProgressBar.update()
    SimilarityProgressMessage.config(text="Calculations Complete!")
    SimilarityProgressBar.destroy()
    gui.update_idletasks()

def RunAllModules():
    ### Will run Acoustic Features, Syntax, and Timing -- will combine all outputs into one csv and into one output folder with figures too
    global RunAll_InputFileDisplay
    global RunAll_OutputFileDisplay
    global RunAll_BirdID_Entry
    global All_Bird_ID
    All_Bird_ID = RunAll_BirdID_Entry.get()
    # Create output folder #
    global RunAll_OutputDir
    RunAll_OutputDir = RunAll_OutputFileDisplay.cget("text")+"/RunAll_Output/"
    try:
        os.makedirs(RunAll_OutputDir)
    except:
        pass
    try:
        RunAllMessage.destroy()
    except:
        pass

    global RunAll_StartRow
    RunAllMessage = tk.Label(RunAllTab, text="")
    RunAllMessage.grid(row=RunAll_StartRow+9, column=0, columnspan=2)
    global RunAllLoadingBar
    RunAllLoadingBar = ttk.Progressbar(RunAllTab, mode="indeterminate")
    RunAllLoadingBar.grid(row=RunAll_StartRow+10, column=0, columnspan=2)
    RunAllLoadingBar.start()
    gui.update_idletasks()
    RunAllMessage.config(text="Calculating Acoustics...")
    RunAllMessage.update()
    Acoustics_Syllables(RunAll=True)
    RunAllMessage.config(text="Calculating Syntax...")
    RunAllMessage.update()
    Syntax(RunAll=True)
    RunAllMessage.config(text="Calculating Timing...")
    RunAllMessage.update()
    Timing(RunAll=True)
    TimingRhythm(RunAll=True)
    RunAllLoadingBar.destroy()
    RunAllMessage.config(text="Complete!")
    RunAllMessage.update()

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
HomeNotebook.add(MoreInfoTab, text="Features")
LinksTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
HomeNotebook.add(LinksTab, text="Helpful Links")

# Welcome Tab #
WelcomeMessage = tk.Label(WelcomeTab, text="Welcome to the AVN GUI! \n \n If this is your first time using the GUI,\n please "
                                           "refer to the \"Features\" tab \n to get started.", font=("Arial", 20)).grid()

# More Info Tab #
MoreInfo_textwrap = 880
MoreInfo_titlefont = ("Arial",10,"bold")
MoreInfo_bodyfont = ("Arial",10)
More_Info_Title = tk.Label(MoreInfoTab, text="What does this GUI do?", font=("Arial", 20,"bold")).grid(row=0)
More_Info_GettingStarted_Title = tk.Label(MoreInfoTab, text="Getting Started", font=MoreInfo_titlefont).grid(row=1)
More_Info_GettingStarted_Body = tk.Label(MoreInfoTab, text="To get the most out of this GUI, you will want to have your zebra finch song recording data "
                                            "organized such that each timepoint you want to analyze has all corresponding song recordings in a single folder. "
                                            "You will also need segmentations of the files in the folder to use many of the features of this GUI. Those segmentations "
                                            "can be generated using your preferred method, so long as they take the form of a .csv file with one row per syllable, "
                                            "with columns: ‘file’ containing the name of the file containing the syllable, ‘onset’ containing the onset time of the "
                                            "syllable within the song file in seconds, and ‘offsets’ containing the offset time of the offset time of the syllable in seconds. "
                                            "We recommend using WhisperSeg to generate these segmentations:",font=MoreInfo_bodyfont,justify="left", wraplength=MoreInfo_textwrap).grid(row=2)
More_Info_GettingStarted_Hyperlink = tk.Label(MoreInfoTab, text="https://github.com/nianlonggu/WhisperSeg", fg="blue", cursor="hand2")
More_Info_GettingStarted_Hyperlink.grid(row=3)
More_Info_GettingStarted_Hyperlink.bind("<Button-1>", lambda e: wb.open_new_tab("https://github.com/nianlonggu/WhisperSeg"))
More_Info_Labeling_Title = tk.Label(MoreInfoTab, text="\nLabeling", font=MoreInfo_titlefont,justify="left", wraplength=MoreInfo_textwrap).grid(row=4)
More_Info_Labeling_Body = tk.Label(MoreInfoTab, text="Labeling: This tab generates syllable labels through UMAP and HDBSCAN clustering based on the syllable segments in the provided segmentation table. "
                                            "It outputs a new copy of the segmentation table called '[Bird_ID]_labels.csv' with labels in a new 'labels' column and "
                                            "UMAP coordinates of each syllable in the ‘X’ and ‘Y’ columns. This labeling table is necessary for many downstream analyses, "
                                            "including ‘Acoustic Features’, ‘Syntax’, and ‘Timing’. The ‘Generate Spectrograms’ tab lets you generate example "
                                            "spectrograms with syllable labels overlaid for visual inspection of annotations.", font=MoreInfo_bodyfont,justify="left", wraplength=MoreInfo_textwrap).grid(row=5)
More_Info_Acoustics_Title = tk.Label(MoreInfoTab, text="\nAcoustic Features", font=MoreInfo_titlefont,justify="left", wraplength=700).grid(row=6)
More_Info_Acoustics_Body = tk.Label(MoreInfoTab, text="Acoustic Features: This tabs lets you calculate acoustic features for syllables in your _labels.csv file, or for entire song files. "
                                            "The ‘Multiple Syllables’ tab allows you to calculate acoustics features for all song files listed in a _labels.csv file. The ‘Single Song Interval’ tab lets you calculate acoustic features "
                                            "over an entire song file, a custom time range within a song file, or a folder of song files. You can choose to save these either as continuous "
                                            "traces, with a value for every spectrogram bin, or as a single average value.", font=MoreInfo_bodyfont,justify="left", wraplength=MoreInfo_textwrap).grid(row=7)
More_Info_Syntax_Title = tk.Label(MoreInfoTab, text="\nSyntax", font=MoreInfo_titlefont,justify="left", wraplength=MoreInfo_textwrap).grid(row=8)
More_Info_Syntax_Body = tk.Label(MoreInfoTab, text="Syntax Analysis: This tab calculates the syllable "
                                                   "order and song structure, and generates heatmaps and "
                                                   "raster plots for visualizing the order of motif syllables", font=MoreInfo_bodyfont,justify="left", wraplength=MoreInfo_textwrap).grid(row=9)
More_Info_Timing_Title = tk.Label(MoreInfoTab, text="\nTiming", font=MoreInfo_titlefont,justify="left", wraplength=MoreInfo_textwrap).grid(row=10)
More_Info_Timing_Body = tk.Label(MoreInfoTab, text="This tab calcualtes the timing of song, including syllable durations and gap durations. " \
                                                    "Also can be utilized to calculate peak frequencies of a song and plot a rhythm " \
                                                    "spectrogram", font=MoreInfo_bodyfont,justify="left", wraplength=MoreInfo_textwrap).grid(row=11)

# Helpful Links Tab #
AVN_Documentation = tk.Label(LinksTab, text="AVN Documentation", fg="blue", cursor="hand2")
AVN_Documentation.grid(row=1)
AVN_Documentation.bind("<Button-1>", lambda e: wb.open_new_tab("https://avn.readthedocs.io/en/latest/modules.html"))
ThereseEmail = tk.Label(LinksTab, text="Email me", fg="blue", cursor="hand2")
ThereseEmail.grid(row=2)
ThereseEmail.bind("<Button-1>", lambda e: wb.open_new_tab("mailto:therese.koch1@gmail.com"))
PreprintLink = tk.Label(LinksTab, text="AVN Preprint", fg="blue", cursor="hand2")
PreprintLink.grid(row=3)
PreprintLink.bind("<Button-1>", lambda e: wb.open_new_tab("https://www.biorxiv.org/content/10.1101/2024.05.10.593561v1"))

# Apply inputs globally #
GlobalInputs = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
HomeNotebook.add(GlobalInputs, text="Global Inputs")
GlobalInputs_Label = tk.Label(GlobalInputs, text="Want to run through the GUI manually using the same inputs "
                                                 "(labeling file, song directory, output directory)? Below you "
                                                 "can apply the same inputs across all AVN modules:", justify="center").grid(row=0)
GlobalInputsFrame = tk.Frame(GlobalInputs,highlightbackground="black", highlightthickness=1)
GlobalInputsFrame.grid(row=1)
GlobalInputs_Input_Button = tk.Button(GlobalInputsFrame, text="Select Labeling File", command=lambda:FileExplorer("GlobalInputs","Input")).grid(row=0, column=0)
GlobalInputs_Input = tk.Label(GlobalInputsFrame, text=Dir_Width*" ", bg="light grey")
GlobalInputs_Input.grid(row=0, column=1, sticky='w', padx=Padding_Width)
GlobalInputs_SongDir_Button = tk.Button(GlobalInputsFrame, text="Select Song Directory", command=lambda:FileExplorer("GlobalInputs","Songs")).grid(row=1, column=0)
GlobalInputs_SongDir = tk.Label(GlobalInputsFrame, text=Dir_Width*" ", bg="light grey")
GlobalInputs_SongDir.grid(row=1, column=1, sticky="w", padx=Padding_Width)
GlobalInputs_BirdID_Label = tk.Label(GlobalInputsFrame, text="Bird ID: ").grid(row=2, column=0)
GlobalInputs_BirdID = tk.Entry(GlobalInputsFrame, font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
GlobalInputs_BirdID.insert(0, "Bird ID")
GlobalInputs_BirdID.bind("<FocusIn>", focus_in)
GlobalInputs_BirdID.bind("<FocusOut>", focus_out)
GlobalInputs_BirdID.grid(row=2, column=1, sticky="w", padx=Padding_Width)
GlobalInputs_Output_Button = tk.Button(GlobalInputsFrame, text="Select Output Directory", command=lambda:FileExplorer("GlobalInputs","Output")).grid(row=3, column=0)
GlobalInputs_Output = tk.Label(GlobalInputsFrame, text=Dir_Width*" ", bg="light grey")
GlobalInputs_Output.grid(row=3, column=1, sticky="w", padx=Padding_Width)


def ApplyGlobalInputs():
    LabelingFile = GlobalInputs_Input.cget("text")
    SongDir = GlobalInputs_SongDir.cget('text')
    Bird_ID = GlobalInputs_BirdID.get()
    OutputDir = GlobalInputs_Output.cget('text')

    Inputs = [LabelingFileDisplay,AcousticsFileDisplay,MultiAcousticsInputDir,SyntaxFileDisplay,Syntax_HeatmapTab_Input,Syntax_Raster_Input,TimingInput_Text,RunAll_InputFileDisplay]
    SongDirs = [FindSongPath,MultiAcousticsSongPath,SyntaxSongLocation,Syntax_HeatmapTab_SongLocation,Syntax_Raster_SongLocation,TimingSongPath,RhythmSpectrogram_InputButton,RunAll_SongPath]
    Bird_IDs = [BirdID_Labeling,LabelingBirdIDText2,MultiAcousticsBirdID,SyntaxBirdID,Syntax_HeatmapTab_BirdID,Syntax_Raster_BirdID,Timing_BirdID,RhythmSpectrogram_BirdID,RunAll_BirdID_Entry]
    Outputs = [LabelingUMAP_OutputText,AcousticsOutput_Text,MultiAcousticsOutputDisplay,SyntaxOutputDisplay,Syntax_HeatmapTab_Output,Syntax_Raster_Output,TimingOutput_Text,RhythmSpectrogram_Output,RunAll_OutputFileDisplay]
    # Apply to all modules #

    for input in Inputs:
        input.config(text=LabelingFile)
        input.update()
    for dir in SongDirs:
        dir.config(text=SongDir)
        dir.update()
    for id in Bird_IDs:
        id.delete(0,END)
        id.insert(0,Bird_ID)
        id.config(fg="black")
        id.update()
    for dir in Outputs:
        dir.config(text=OutputDir)
        dir.update()

GlobalInputs_Submit_Button = tk.Button(GlobalInputs, text="Apply Inputs", command=lambda:ApplyGlobalInputs()).grid(row=10, column=0, columnspan=2)


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
# LabelingNotebook.add(LabelingBulkSave, text="Save Many Spectrograms")

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
Labeling_SpectrogramCheck.grid(row=7, column=0, columnspan=2)

Labeling_Padding = tk.Label(LabelingMainFrame, text="", font=("Arial", 20)).grid(row=5)
Label_CheckLabel = tk.Label(LabelingMainFrame, text="Number of files to visually inspect:").grid(row=6, column=0)
global Label_CheckNumber
Label_CheckNumber = StringVar()
Label_CheckNumber.set("5")
Labeling_SpectrogramCheckNumber = tk.Spinbox(LabelingMainFrame,from_=1, to=99, textvariable=Label_CheckNumber, width=15)
Labeling_SpectrogramCheckNumber.grid(row=6, column=1, sticky="W",padx=Padding_Width)

Labeling_Padding2 = tk.Label(LabelingMainFrame, text="", font=("Arial", 20)).grid(row=8)

LabelingButton = tk.Button(LabelingMainFrame, text="Run", command = lambda : Labeling())
LabelingButton.grid(row=8, column=0, columnspan=2)

global LabelingBirdIDText
LabelingBirdIDText = tk.Entry(LabelingMainFrame, font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
LabelingBirdIDText.insert(0, "Bird ID")
LabelingBirdIDText.bind("<FocusIn>", focus_in)
LabelingBirdIDText.bind("<FocusOut>", focus_out)
LabelingBirdIDText.grid(row=3, column=1, sticky="w", padx=Padding_Width)

LabelingBirdIDLabel = tk.Label(LabelingMainFrame, text="Bird ID:")
LabelingBirdIDLabel.grid(row=3, column=0)

global LabelingDirectoryText
LabelingDirectoryText = tk.Label(LabelingMainFrame, text=Dir_Width*" ", bg="light grey", anchor="w")
LabelingDirectoryText.grid(row=1, column=1, columnspan=2, sticky="w", padx=Padding_Width)
LabelingFileExplorer = tk.Button(LabelingMainFrame, text="Select Segmentation File", command=lambda: FileExplorer("Labeling", "Input"))
LabelingFileExplorer.grid(row=1, column=0)

global LabelingSongLocation
LabelingSongLocation = tk.Label(LabelingMainFrame, text=Dir_Width*" ", bg="light grey", anchor="w")
LabelingSongLocation.grid(row=2, column=1, sticky="w",padx=Padding_Width)
LabelingSongLocation_Button = tk.Button(LabelingMainFrame,text="Select Song Directory", command=lambda:FileExplorer("Labeling","Songs"))
LabelingSongLocation_Button.grid(row=2, column=0)

global LabelingOutputFile_Text
LabelingOutputFile_Text = tk.Label(LabelingMainFrame, text=Dir_Width*" ", bg="light grey", anchor="w")
LabelingOutputFile_Text.grid(row=4, column=1, columnspan=2, sticky="w", padx=Padding_Width)

def LabelingOutputDirectory():
    global LabelingOutputFile_Text
    Selected_Output = filedialog.askdirectory()
    LabelingOutputFile_Text.config(text=str(Selected_Output))
    LabelingOutputFile_Text.update()

LabelingOutputFile_Button = tk.Button(LabelingMainFrame, text="Select Output Directory", command=lambda:FileExplorer("Labeling", "Output"))
LabelingOutputFile_Button.grid(row=4, column=0)

# Labeling Spectrogram and UMAP Generation
global LabelingFileDisplay
LabelingFileDisplay = tk.Label(LabelingSpectrogram_UMAP, text=Dir_Width*" ", bg="light grey")
LabelingFileDisplay.grid(row=0, column=1, sticky="w", padx=Padding_Width)
FindLabelingFile_Button = tk.Button(LabelingSpectrogram_UMAP, text="Select Labeling File", command=lambda:FileExplorer("Labeling_UMAP", "Input"))
FindLabelingFile_Button.grid(row=0, column=0)

FindSongPath_Button = tk.Button(LabelingSpectrogram_UMAP, text="Select Song Directory", command=lambda:FileExplorer("Labeling_UMAP", "Songs"))
FindSongPath_Button.grid(row=1, column=0)
global FindSongPath
FindSongPath = tk.Label(LabelingSpectrogram_UMAP, text=Dir_Width*" ", bg="light grey")
FindSongPath.grid(row=1, column=1, sticky="w", padx=Padding_Width)

BirdIDText_Labeling = tk.Label(LabelingSpectrogram_UMAP, text="Bird ID:")
BirdIDText_Labeling.grid(row=2, column=0)
global BirdID_Labeling
BirdID_Labeling = tk.Entry(LabelingSpectrogram_UMAP,font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
BirdID_Labeling.insert(0, "Bird ID")
BirdID_Labeling.bind("<FocusIn>", focus_in)
BirdID_Labeling.bind("<FocusOut>", focus_out)
BirdID_Labeling.grid(row=2, column=1, sticky="w", padx=Padding_Width)
LabelingUMAP_OutputButton = tk.Button(LabelingSpectrogram_UMAP, text="Select Output Directory", command=lambda:FileExplorer("Labeling_UMAP", "Output"))
LabelingUMAP_OutputButton.grid(row=3, column=0)
global LabelingUMAP_OutputText
LabelingUMAP_OutputText = tk.Label(LabelingSpectrogram_UMAP, text=Dir_Width*" ", bg="light grey")
LabelingUMAP_OutputText.grid(row=3, column=1, sticky="w", padx=Padding_Width)

global MaxFiles_Labeling
MaxFiles_Labeling_Text = tk.Label(LabelingSpectrogram_UMAP, text="Number of Spectrograms to Generate:").grid(row=4, column=0)
MaxFiles_Labeling = tk.Spinbox(LabelingSpectrogram_UMAP, from_=0, to=99, font=("Arial, 10"), width=10, justify="center")
MaxFiles_Labeling.grid(row=4, column=1, sticky="w", padx=Padding_Width)

GenerateSpectrogram_Button = tk.Button(LabelingSpectrogram_UMAP, text="Generate Spectrograms", command=lambda:LabelingSpectrograms())
GenerateSpectrogram_Button.grid(row=5, column=0, columnspan=2)

# Labeling Bulk Save
LabelingBirdIDText2 = tk.Entry(LabelingBulkSave, font=("Arial", 15), justify="center", fg="grey", width=BirdID_Width)
LabelingBirdIDText2.insert(0, "Bird ID")
LabelingBirdIDText2.bind("<FocusIn>", focus_in)
LabelingBirdIDText2.bind("<FocusOut>", focus_out)
LabelingBirdIDText2.grid(row=2, column=1, sticky="w", padx=Padding_Width)

LabelingBirdIDLabel2 = tk.Label(LabelingBulkSave, text="Bird ID:")
LabelingBirdIDLabel2.grid(row=2, column=0)
LabelingFileExplorer2 = tk.Button(LabelingBulkSave, text="Select Labeling File",command=lambda: LabelingFileExplorerFunctionSaving())
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
               min_level_db_entry_Labeling,n_fft_entry_Labeling,win_length_entry_Labeling,hop_length_entry_Labeling]
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

# max_spec_size_Labeling_Error = tk.Label(LabelingSettingsFrame, text="", font=ErrorFont)
# max_spec_size_Labeling_Error.grid(row=17, column=1)
# max_spec_size_text_Labeling = tk.Label(LabelingSettingsFrame, text="max_spec_size").grid(row=16,column=0)
# max_spec_size_entry_Labeling = tk.Entry(LabelingSettingsFrame, justify="center")
# max_spec_size_entry_Labeling.insert(0, "300")
# max_spec_size_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(max_spec_size_entry_Labeling,'max_spec_size_entry_Labeling',max_spec_size_Labeling_Error))
# max_spec_size_entry_Labeling.grid(row=16, column=1)
# max_spec_size_reset_Labeling = tk.Button(LabelingSettingsFrame, text="Reset",command=lambda: ResetLabelingSetting(max_spec_size_entry_Labeling, LabelSpecList,max_spec_size_Labeling_Error)).grid(row=16, column=2)
# max_spec_size_moreinfo_Labeling = tk.Button(LabelingSettingsFrame, text='?', state="disabled", fg="black")
# max_spec_size_moreinfo_Labeling.grid(row=16, column=3, sticky=W)
# max_spec_size_moreinfo_Labeling.bind("<Enter>", MoreInfo)
# max_spec_size_moreinfo_Labeling.bind("<Leave>", LessInfo)

LabelSpecList = [bandpass_lower_cutoff_entry_Labeling,bandpass_upper_cutoff_entry_Labeling,a_min_entry_Labeling,ref_db_entry_Labeling,
                 min_level_db_entry_Labeling,n_fft_entry_Labeling,win_length_entry_Labeling,hop_length_entry_Labeling]
LabelingSpectrogram_ResetAllSettings = tk.Button(LabelingSettingsFrame, text="Reset All", command=lambda:ResetLabelingSetting("all", LabelSpecList))
LabelingSpectrogram_ResetAllSettings.grid(row=25, column=1)

global LabelErrorLabels
LabelErrorLabels = [bandpass_lower_cutoff_Labeling_Error,bandpass_upper_cutoff_Labeling_Error,a_min_Labeling_Error,ref_db_Labeling_Error,min_level_db_Labeling_Error,n_fft_Labeling_Error,win_length_Labeling_Error,hop_length_Labeling_Error]

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
global metric_variable
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
def ResetLabelingClusterSetting(Setting):
    DefaultValues = [0.04, 5]
    if Setting == "min_cluster":
        min_cluster_prop_entry_Labeling.delete(0,END)
        min_cluster_prop_entry_Labeling.insert(0,"0.04")
        min_cluster_prop_entry_Labeling.config(bg="white")
        min_cluster_prop_entry_Labeling.update()
        min_cluster_prop_Labeling_Error.config(text="")
        min_cluster_prop_Labeling_Error.update()
    elif Setting == "min_samples":
        min_samples_entry_Labeling.delete(0, END)
        min_samples_entry_Labeling.insert(0,"5")
        min_samples_entry_Labeling.config(bg="white")
        min_samples_entry_Labeling.update()
        min_samples_entry_Labeling.config(text="")
        min_samples_entry_Labeling.update()
    elif Setting == "all":
        min_cluster_prop_entry_Labeling.delete(0, END)
        min_cluster_prop_entry_Labeling.insert(0,"0.04")
        min_cluster_prop_entry_Labeling.config(bg="white")
        min_cluster_prop_entry_Labeling.update()
        min_cluster_prop_Labeling_Error.config(text="")
        min_cluster_prop_Labeling_Error.update()

        min_samples_entry_Labeling.delete(0, END)
        min_samples_entry_Labeling.insert(0,"5")
        min_samples_entry_Labeling.config(bg="white")
        min_samples_entry_Labeling.update()
        min_samples_entry_Labeling.config(text="")
        min_samples_entry_Labeling.update()


min_cluster_prop_Labeling_Error = tk.Label(LabelingClusterSettings, text="", font=ErrorFont)
min_cluster_prop_Labeling_Error.grid(row=1, column=1)
min_cluster_prop_text_Labeling = tk.Label(LabelingClusterSettings, text="min_cluster_prop").grid(row=0, column=0)
min_cluster_prop_entry_Labeling = tk.Entry(LabelingClusterSettings, justify="center")
min_cluster_prop_entry_Labeling.insert(0, "0.04")
min_cluster_prop_entry_Labeling.bind("<FocusOut>", lambda value:Validate_Settings(min_cluster_prop_entry_Labeling,'min_cluster_prop_entry_Labeling',min_cluster_prop_Labeling_Error))
min_cluster_prop_entry_Labeling.grid(row=0, column=1)
min_cluster_prop_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(Setting="min_cluster_prop")).grid(row=0, column=2)
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
min_samples_reset_Labeling = tk.Button(LabelingClusterSettings, text="Reset",command=lambda: ResetLabelingClusterSetting(Setting="min_samples")).grid(row=2, column=2)
min_samples_moreinfo_Labeling = tk.Button(LabelingClusterSettings, text='?', state="disabled", fg="black")
min_samples_moreinfo_Labeling.grid(row=2, column=3, sticky=W)
min_samples_moreinfo_Labeling.bind("<Enter>", MoreInfo)
min_samples_moreinfo_Labeling.bind("<Leave>", LessInfo)

LabelingClusterVars = [min_cluster_prop_entry_Labeling,spread_entry_Labeling]
LabelingClusterResetAll = tk.Button(LabelingClusterSettings, text="Reset All", command=lambda: ResetLabelingClusterSetting(Setting="all"))
LabelingClusterResetAll.grid(row=4, column=1)

LabelingClusterDialog = tk.Label(LabelingClusterSettings, text="")
LabelingClusterDialog.grid(row=1, column=4, rowspan=10)
LabelingClusterTitle = tk.Label(LabelingClusterSettings, text="", font=("Arial", 20, "bold"), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
LabelingClusterTitle.grid(row=0, column=4)

LabelingClusterPadding = tk.Label(LabelingClusterSettings, pady=10, font=("Arial",25))
LabelingClusterPadding.grid(row=5, column=1)

LabelingClusterVars = [min_cluster_prop_entry_Labeling,min_samples_entry_Labeling]

def LoadSettings_Labeling(Tab):
    if Tab == "Spectro":
        SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        if SettingsMetadata != "":
            SettingsMetadata_df = pd.read_csv(SettingsMetadata)

            bandpass_lower_cutoff_entry_Labeling.delete(0,END)
            bandpass_lower_cutoff_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling"][1]))
            bandpass_lower_cutoff_entry_Labeling.config(bg="white")
            bandpass_lower_cutoff_entry_Labeling.update()
            bandpass_lower_cutoff_Labeling_Error.config(text="")

            bandpass_upper_cutoff_entry_Labeling.delete(0, END)
            bandpass_upper_cutoff_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.1"][1]))
            bandpass_upper_cutoff_entry_Labeling.config(bg="white")
            bandpass_upper_cutoff_entry_Labeling.update()
            bandpass_upper_cutoff_Labeling_Error.config(text="")

            a_min_entry_Labeling.delete(0, END)
            a_min_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.2"][1]))
            a_min_entry_Labeling.config(bg="white")
            a_min_entry_Labeling.update()
            a_min_Labeling_Error.config(text="")

            ref_db_entry_Labeling.delete(0, END)
            ref_db_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.3"][1]))
            ref_db_entry_Labeling.config(bg="white")
            ref_db_entry_Labeling.update()
            ref_db_Labeling_Error.config(text="")

            min_level_db_entry_Labeling.delete(0, END)
            min_level_db_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.4"][1]))
            min_level_db_entry_Labeling.config(bg='white')
            min_level_db_entry_Labeling.update()
            min_level_db_Labeling_Error.config(text='')

            n_fft_entry_Labeling.delete(0, END)
            n_fft_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.5"][1]))
            n_fft_entry_Labeling.config(bg="white")
            n_fft_entry_Labeling.update()
            n_fft_Labeling_Error.config(text='')

            win_length_entry_Labeling.delete(0, END)
            win_length_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.6"][1]))
            win_length_entry_Labeling.config(bg="white")
            win_length_entry_Labeling.update()
            win_length_Labeling_Error.config(text='')

            hop_length_entry_Labeling.delete(0, END)
            hop_length_entry_Labeling.insert(0, str(SettingsMetadata_df["Labeling.7"][1]))
            hop_length_entry_Labeling.config(bg='white')
            hop_length_entry_Labeling.update()
            hop_length_Labeling_Error.config(text='')

            # max_spec_size_entry_Labeling.delete(0, END)
            # max_spec_size_entry_Labeling.insert(0, str(SettingsMetadata_df["Value"][8]))
    if Tab == "UMAP":
        SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        if len(SettingsMetadata) > 0:
            SettingsMetadata_df = pd.read_csv(SettingsMetadata)

            n_neighbors_entry_Labeling.delete(0, END)
            n_neighbors_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP"][1]))
            n_neighbors_entry_Labeling.config(bg='white')
            n_neighbors_entry_Labeling.update()
            n_neighbors_Labeling_Error.config(text='')

            n_components_entry_Labeling.delete(0, END)
            n_components_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP.1"][1]))
            n_components_entry_Labeling.config(bg='white')
            n_components_entry_Labeling.update()
            n_components_Labeling_Error.config(text='')

            min_dist_entry_Labeling.delete(0, END)
            min_dist_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP.2"][1]))
            min_dist_entry_Labeling.config(bg='white')
            min_dist_entry_Labeling.update()
            min_dist_Labeling_Error.config(text='')

            spread_entry_Labeling.delete(0, END)
            spread_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP.3"][1]))
            spread_entry_Labeling.config(bg='white')
            spread_entry_Labeling.update()
            spread_Labeling_Error.config(text='')

            global metric_variable
            metric_variable.set('euclidean')
            metric_Labeling_Error.config(text='')

            random_state_entry_Labeling.delete(0, END)
            if str(SettingsMetadata_df["UMAP.5"][1]) == "nan":
                random_state_entry_Labeling.insert(0, "None")
            else:
                random_state_entry_Labeling.insert(0, str(SettingsMetadata_df["UMAP.5"][1]))
            random_state_entry_Labeling.config(bg='white')
            random_state_entry_Labeling.update()
            random_state_Labeling_Error.config(text='')

    if Tab == "Cluster":
        SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
        if len(SettingsMetadata) > 0:
            SettingsMetadata_df = pd.read_csv(SettingsMetadata)

            min_cluster_prop_entry_Labeling.delete(0, END)
            min_cluster_prop_entry_Labeling.insert(0, str(SettingsMetadata_df["Clustering"][1]))
            min_cluster_prop_entry_Labeling.update()

            min_samples_entry_Labeling.delete(0, END)
            min_samples_entry_Labeling.insert(0, str(SettingsMetadata_df["Clustering.1"][1]))
            min_samples_entry_Labeling.update()

LoadLabelingSettings_Spectro = tk.Button(LabelingSettingsFrame, text="Load Settings", command=lambda: LoadSettings_Labeling("Spectro"))
LoadLabelingSettings_Spectro.grid(row=25, column=0)

LoadLabelingSettings_UMAP = tk.Button(LabelingUMAPSettings, text="Load Settings", command=lambda: LoadSettings_Labeling("UMAP"))
LoadLabelingSettings_UMAP.grid(row=25, column=0)

LoadLabelingSettings_Cluster = tk.Button(LabelingClusterSettings, text="Load Settings", command=lambda: LoadSettings_Labeling("Cluster"))
LoadLabelingSettings_Cluster.grid(row=4, column=0)

### Single File Acoustic Features Window ###

AcousticsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
notebook.add(AcousticsFrame, text="Acoustic Features")


AcousticsMainFrameSingle = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
AcousticsNotebook = ttk.Notebook(AcousticsFrame)
AcousticsNotebook.grid(row=1)
AcousticsMainFrameMulti = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
AcousticsNotebook.add(AcousticsMainFrameMulti, text="Multiple Syllables")
AcousticsNotebook.add(AcousticsMainFrameSingle, text="Single Song Interval")

AcousticsInputLabel = tk.Label(AcousticsMainFrameSingle, text="Input Type:").grid(row=0, column=0)
global AcousticsInputVar
AcousticsInputVar = IntVar()
AcousticsInputType_Single = tk.Radiobutton(AcousticsMainFrameSingle, text="Single File", variable=AcousticsInputVar, value=0).grid(row=0, column=1)
AcousticsInputType_Folder = tk.Radiobutton(AcousticsMainFrameSingle, text="Whole Folder", variable=AcousticsInputVar, value=1).grid(row=0, column=2)


AcousticsFileExplorer = tk.Button(AcousticsMainFrameSingle, text="Select Song Directory",
                                  command=lambda: FileExplorer("Acoustics_Single", "Input"))
AcousticsFileExplorer.grid(row=1, column=0)
AcousticsFileDisplay = tk.Label(AcousticsMainFrameSingle, text=2*Dir_Width*" ", bg="light grey", anchor=tk.W)
AcousticsFileDisplay.grid(row=1, column=1, padx=Padding_Width, columnspan=2)
#
# SingleAcousticsBirdID_Label = tk.Label()
# global SingleAcousticsBirdID
# SingleAcousticsBirdID = tk.Entry(AcousticsMainFrameSingle, text="Bird ID: ").grid(row=2, column=0)

AcousticsOutput_Button = tk.Button(AcousticsMainFrameSingle, text="Select Output Directory",
                                   command=lambda: FileExplorer("Acoustics_Single", "Output"))
AcousticsOutput_Button.grid(row=2, column=0)
global AcousticsOutput_Text
AcousticsOutput_Text = tk.Label(AcousticsMainFrameSingle, text=2*Dir_Width*" ", bg="light grey")
AcousticsOutput_Text.grid(row=2, column=1, padx=Padding_Width, columnspan=2)

Calculation_Method_Label = tk.Label(AcousticsMainFrameSingle, text="Calculation Method:").grid(row=3, column=0, padx=Padding_Width)
global ContCalc
ContCalc = IntVar()
ContCalc.set(0)
Continuous_Calculations = tk.Radiobutton(AcousticsMainFrameSingle, text="Continuous Calcualtions",
                                         variable=ContCalc, value=0).grid(row=3, column=1, columnspan=1)
Average_Calculations = tk.Radiobutton(AcousticsMainFrameSingle, text="Average Calcualtions", variable=ContCalc,
                                      value=1).grid(row=3, column=2, columnspan=1)
CalculationMethod_Info = tk.Button(AcousticsMainFrameSingle, text="?", state=DISABLED)
CalculationMethod_Info.grid(row=3, column=3, sticky="W")
CalculationMethod_Info.bind("<Enter>", MoreInfo)
CalculationMethod_Info.bind("<Leave>", LessInfo)

AcousticsCalcTitle=tk.Label(AcousticsMainFrameSingle, text="", font=("Arial", 15, "bold"))
AcousticsCalcTitle.grid(row=0, column=5)
AcousticsCalcMethod=tk.Label(AcousticsMainFrameSingle, text="")
AcousticsCalcMethod.grid(row=1, column=5, rowspan=2)

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

AcousticsPadding = tk.Label(AcousticsMainFrameSingle, text=" ").grid(row=StartRow+11)

### Multiple Syllable Acoustic Features Window ###

# AcousticsMainFrameMulti = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
# AcousticsNotebook.add(AcousticsMainFrameMulti, text="Multiple Syllables")
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

MultiAcousticsFileExplorer = tk.Button(AcousticsMainFrameMulti, text="Select Labeling File",command=lambda: FileExplorer("Acoustics_Multi", "Input"))
MultiAcousticsFileExplorer.grid(row=0, column=0)
global MultiAcousticsInputDir
MultiAcousticsInputDir = tk.Label(AcousticsMainFrameMulti, text=Dir_Width*" ", bg="light grey")
MultiAcousticsInputDir.grid(row=0, column=1, padx=Padding_Width, sticky="w")

global MultiAcousticsSongPath
MultiAcousticsSongPath_Button = tk.Button(AcousticsMainFrameMulti, text="Select Song Directory",command=lambda: FileExplorer("Acoustics_Multi", "Songs"))
MultiAcousticsSongPath_Button.grid(row=1, column=0)
MultiAcousticsSongPath = tk.Label(AcousticsMainFrameMulti, text=Dir_Width*" ", bg="light grey")
MultiAcousticsSongPath.grid(row=1, column=1, sticky="w", padx=Padding_Width)

global MultiAcousticsBirdID
MultiAcousticsBirdID = tk.Entry(AcousticsMainFrameMulti, font=("Arial", 15), justify="center", fg="grey",
                               width=BirdID_Width)
MultiAcousticsBirdID.insert(0, "Bird ID")
MultiAcousticsBirdID.bind("<FocusIn>", focus_in)
MultiAcousticsBirdID.bind("<FocusOut>", focus_out)
MultiAcousticsBirdID.grid(row=2, column=1, sticky="w", padx=Padding_Width)
MultiAcousticsBirdIDLabel = tk.Label(AcousticsMainFrameMulti, text="Bird ID: ").grid(row=2, column=0)

MultiAcousticsOutputButton = tk.Button(AcousticsMainFrameMulti, text="Select Output Directory", command=lambda: FileExplorer("Acoustics_Multi", "Output"))
MultiAcousticsOutputButton.grid(row=3, column=0)
global MultiAcousticsOutputDisplay
MultiAcousticsOutputDisplay = tk.Label(AcousticsMainFrameMulti, text=Dir_Width*" ", bg="light grey")
MultiAcousticsOutputDisplay.grid(row=3, column=1, padx=Padding_Width, sticky="w")
MultiAcousticsPadding = tk.Label(AcousticsMainFrameMulti, text=" ").grid(row=4)
MultiAcousticsText = tk.Label(AcousticsMainFrameMulti, text="Acoustic features to analyze:", justify="center")
MultiAcousticsText.grid(row=5, column=1)

def LoadSettings_Acoustic():
    SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
    if SettingsMetadata != "":
        SettingsMetadata_df = pd.read_csv(SettingsMetadata)
        win_length_entry_Acoustics.delete(0,END)
        win_length_entry_Acoustics.insert(0, str(SettingsMetadata_df["win_length"][0]))

        hop_length_entry_Acoustics.delete(0, END)
        hop_length_entry_Acoustics.insert(0, str(SettingsMetadata_df["hop_length"][0]))

        n_fft_entry_Acoustics.delete(0, END)
        n_fft_entry_Acoustics.insert(0, str(SettingsMetadata_df["n_fft"][0]))

        max_F0_entry_Acoustics.delete(0, END)
        max_F0_entry_Acoustics.insert(0, str(SettingsMetadata_df["max_F0"][0]))

        min_frequency_entry_Acoustics.delete(0, END)
        min_frequency_entry_Acoustics.insert(0, str(SettingsMetadata_df["min_frequency"][0]))

        freq_range_entry_Acoustics.delete(0, END)
        freq_range_entry_Acoustics.insert(0, str(SettingsMetadata_df["freq_range"][0]))

        baseline_amp_entry_Acoustics.delete(0, END)
        baseline_amp_entry_Acoustics.insert(0, str(SettingsMetadata_df["baseline_amp"][0]))

        fmax_yin_entry_Acoustics.delete(0, END)
        fmax_yin_entry_Acoustics.insert(0, str(SettingsMetadata_df["fmax_yin"][0]))

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

MultiSettings_StartRow = 7
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

### Syntax ###
SyntaxFrame = tk.Frame(gui)
notebook.add(SyntaxFrame, text="Syntax")

SyntaxMainFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SyntaxNotebook = ttk.Notebook(SyntaxFrame)
SyntaxNotebook.grid(row=1)
SyntaxNotebook.add(SyntaxMainFrame, text="Home")
Syntax_HeatmapTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SyntaxNotebook.add(Syntax_HeatmapTab, text="Plot Transition Matrix")
Syntax_RasterTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SyntaxNotebook.add(Syntax_RasterTab, text="Generate Raster Plot")
SyntaxSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SyntaxNotebook.add(SyntaxSettingsFrame, text="Advanced Settings")
# SyntaxInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
# SyntaxNotebook.add(SyntaxInfoFrame, text="Info")

SyntaxFileButton = tk.Button(SyntaxMainFrame, text="Select Labeling File", command=lambda:FileExplorer("Syntax", "Input"))
SyntaxFileButton.grid(row=0,column=0)
global SyntaxFileDisplay
SyntaxFileDisplay = tk.Label(SyntaxMainFrame, text=Dir_Width*" ", bg="light grey")
SyntaxFileDisplay.grid(row=0, column=1, padx=Padding_Width, columnspan=3, sticky='w')

SyntaxSongLocation_Button = tk.Button(SyntaxMainFrame, text="Select Song Directory", command=lambda:FileExplorer("Syntax", "Songs"))
SyntaxSongLocation_Button.grid(row=1, column=0)
global SyntaxSongLocation
SyntaxSongLocation = tk.Label(SyntaxMainFrame, text=Dir_Width*" ", bg="light grey")
SyntaxSongLocation.grid(row=1, column=1, sticky="w", padx=Padding_Width)

SyntaxOutputButton = tk.Button(SyntaxMainFrame, text="Select Output Directory", command=lambda:FileExplorer("Syntax", "Output"))
SyntaxOutputButton.grid(row=3, column=0)
global SyntaxOutputDisplay
SyntaxOutputDisplay = tk.Label(SyntaxMainFrame, text=Dir_Width*" ", bg="light grey")
SyntaxOutputDisplay.grid(row=3, column=1, padx=Padding_Width, columnspan=3, sticky='w')

global DropCalls
DropCalls = IntVar()
DropCallsCheckbox = tk.Checkbutton(SyntaxMainFrame, text= "Drop Calls", variable=DropCalls)
DropCallsCheckbox.grid(row=4, column=0, columnspan=2)

SyntaxRunButton = tk.Button(SyntaxMainFrame, text="Run", command=lambda: Syntax())
SyntaxRunButton.grid(row=5, column=0, columnspan=2)
SyntaxBirdID_Text = tk.Label(SyntaxMainFrame, text="Bird ID:").grid(row=2, column=0)
global SyntaxBirdID
SyntaxBirdID = tk.Entry(SyntaxMainFrame, fg="grey", font=("Arial",15), justify="center", width=BirdID_Width)
SyntaxBirdID.insert(0, "Bird ID")
SyntaxBirdID.bind("<FocusIn>", focus_in)
SyntaxBirdID.bind("<FocusOut>", focus_out)
SyntaxBirdID.grid(row=2, column=1, sticky='w', padx=Padding_Width)

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

def LoadSettings_Syntax():
    SettingsFile = filedialog.askopenfilenames(filetypes=[(".csv files", "*.csv")])
    if len(SettingsFile) > 0:
        SettingsFile_df = pd.read_csv(SettingsFile[0])

        min_gap_entry_Syntax.delete(0, END)
        NewValue = SettingsFile_df["min_gap"][0]
        min_gap_entry_Syntax.insert(0,NewValue)
        min_gap_entry_Syntax.update()

        if str(SettingsFile_df["calls_dropped"][0]) == "TRUE":
            DropCalls.set(1)
        if str(SettingsFile_df["calls_dropped"][0]) == "FALSE":
            DropCalls.set(0)

SyntaxLoadSettings = tk.Button(SyntaxSettingsFrame, text="Load Settings", command=lambda:LoadSettings_Syntax())
SyntaxLoadSettings.grid(row=5)

Syntax_HeatmapTab_Input_Button = tk.Button(Syntax_HeatmapTab, text="Select Labeling File", command=lambda:FileExplorer("SyntaxHeatmap", "Input"))
Syntax_HeatmapTab_Input_Button.grid(row=0, column=0)
global Syntax_HeatmapTab_Input
Syntax_HeatmapTab_Input = tk.Label(Syntax_HeatmapTab, text=Dir_Width*" ", bg="light grey")
Syntax_HeatmapTab_Input.grid(row=0, column=1, sticky="w", padx=Padding_Width)
Syntax_HeatmapTab_BirdID_Label = tk.Label(Syntax_HeatmapTab, text="Bird ID: ").grid(row=1, column=0)
Syntax_HeatmapTab_BirdID = tk.Entry(Syntax_HeatmapTab, fg="grey",font=("Arial",15), justify="center", width=BirdID_Width)
Syntax_HeatmapTab_BirdID.insert(0, "Bird ID")
Syntax_HeatmapTab_BirdID.bind("<FocusIn>", focus_in)
Syntax_HeatmapTab_BirdID.bind("<FocusOut>", focus_out)
Syntax_HeatmapTab_BirdID.grid(row=1, column=1, sticky='w', padx=Padding_Width)
Syntax_HeatmapTab_SongLocation_Button = tk.Button(Syntax_HeatmapTab, text="Select Song Directory", command=lambda:FileExplorer("SyntaxHeatmap", "Song"))
Syntax_HeatmapTab_SongLocation_Button.grid(row=2, column=0)
global Syntax_HeatmapTab_SongLocation
Syntax_HeatmapTab_SongLocation = tk.Label(Syntax_HeatmapTab, text=Dir_Width*" ", bg="light grey")
Syntax_HeatmapTab_SongLocation.grid(row=2, column=1, sticky="w", padx=Padding_Width)
Syntax_HeatmapTab_Output_Button = tk.Button(Syntax_HeatmapTab, text="Select Output Directory", command=lambda:FileExplorer("SyntaxHeatmap", "Output"))
Syntax_HeatmapTab_Output_Button.grid(row=3, column=0)
global Syntax_HeatmapTab_Output
Syntax_HeatmapTab_Output = tk.Label(Syntax_HeatmapTab, text=Dir_Width*" ", bg="light grey")
Syntax_HeatmapTab_Output.grid(row=3, column=1, sticky="w", padx=Padding_Width)

SyntaxHeatmapType_Label = tk.Label(Syntax_HeatmapTab, text="Select Heatmap Type: ").grid(row=6, column=0)
Var_Heatmap = IntVar()
SelectCountHeatmap = tk.Radiobutton(Syntax_HeatmapTab, text="Count Matrix", variable=Var_Heatmap, value=1)
SelectCountHeatmap.grid(row=6, column=1)
SelectProbHeatmap = tk.Radiobutton(Syntax_HeatmapTab, text="Probability Matrix", variable=Var_Heatmap, value=2)
SelectProbHeatmap.grid(row=6, column=2, sticky="w", padx=Padding_Width)

def GenerateMatrixHeatmap():
    global Syntax_HeatmapTab_Input
    global Syntax_HeatmapTab_Output
    syntax_data = Syntax(Mode="Heatmap")
    try: # Clear previous plot, if one exists
        plt.clf()
    except:
        pass
    # global syntax_data
    Bird_ID = Syntax_HeatmapTab_BirdID.get()
    global X_Size
    global Y_Size
    global ChangeFigSize
    if ChangeFigSize.get() == 1:
        FigX = int(X_Size.get())
        FigY = int(X_Size.get())
    else:
        FigX = 10
        FigY = 10
    if Var_Heatmap.get() == 1:
        # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
        # fig = Figure()

        plt.figure(figsize=(FigX,FigY))

        sns.heatmap(syntax_data.trans_mat, annot=True, fmt='0.0f')
        plt.title(str(Bird_ID)+" Count Transition Matrix")
        plt.xticks(rotation=0)
        FileSuffix = "_Count_Transition_Matrix.png"


    if Var_Heatmap.get() == 2:
        # Add option for custom dimensions using "plt.figure(figsize = (x, y))"
        plt.figure(figsize=(FigX,FigY))

        sns.heatmap(syntax_data.trans_mat_prob, annot=True, fmt='0.2f')

        plt.title(str(Bird_ID)+" Probability Transition Matrix")
        plt.xticks(rotation=0)
        FileSuffix = "_Probability_Transition_Matrix.png"

    Output = Syntax_HeatmapTab_Output.cget("text")+"/"
    plt.savefig(Output+Bird_ID+FileSuffix)
    plt.clf()

    # Heatmap_canvas = FigureCanvasTkAgg(fig, master=MatrixHeatmap_Display)  # A tk.DrawingArea.
    # Heatmap_canvas.draw()
    # Heatmap_canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)
    Heatmap = PhotoImage(file=Output+Bird_ID+FileSuffix)
    try:
        MatrixHeatmap_Display_Label.destroy()
        MatrixHeatmap_Display_Label = tk.Label(MatrixHeatmap_Display, image= Heatmap)
        MatrixHeatmap_Display_Label.grid(row=1, column=0, columnspan=3)
    except:
        MatrixHeatmap_Display = tk.Toplevel(gui)
        MatrixHeatmap_Display_Label = tk.Label(MatrixHeatmap_Display, image=Heatmap)
        MatrixHeatmap_Display_Label.grid(row=1, column=0, columnspan=3)

    time.sleep(1)
    # shutil.rmtree(str(SyntaxOutputDisplay.cget("text"))+"/Heatmap.png")

    Type = ""
    if Var_Heatmap.get() == 1:
        Type = "Count"
    if Var_Heatmap.get() == 2:
        Type = "Prob"
    def SaveHeatmap(plt, Type):
        if Type == 1:
            Style = "Count"
        if Type == 2:
            Style = "Probability"
        Bird_ID = Syntax_HeatmapTab_BirdID.get()
        f = asksaveasfile(initialfile=str(Bird_ID)+"_"+Style+"Matrix_Heatmap.png",
                          defaultextension=".png", filetypes=[("All Files", "*.*"), ("PNG Image File", "*.png")])
        files = [('PNG Image', '*.png'), ('All Files', '*.*')]
        if len(str(f)) > 0:
            filename = str(f).split("\'")[1]
            try:
                shutil.rmtree(filename)
            except:
                pass
            plt.savefig(filename)

    MatrixHeatmap_SaveButton = tk.Button(MatrixHeatmap_Display, text="Save", command=lambda:SaveHeatmap(plt,Var_Heatmap.get()))
    MatrixHeatmap_SaveButton.grid(row=0, column=1)
    MatrixHeatmap_Display.mainloop()

MatrixHeatmap_Button = tk.Button(Syntax_HeatmapTab, text="Generate Matrix Heatmap", command=lambda:GenerateMatrixHeatmap())
MatrixHeatmap_Button.grid(row=7, column=0, columnspan=2)

def GenerateRasterPlot():
    try: # Clear previous plot, if one exists
        plt.clf()
    except:
        pass
    global Syntax_Raster_Input
    Bird_ID = Syntax_Raster_BirdID.get()
    try: SyntaxAlignment_WarningMessage.destroy()
    except: pass
    syntax_data = Syntax(Mode="Raster")
    SyntaxOutputFolder = Syntax_Raster_Output.cget("text")+"/"

    if SyntaxAlignmentVar.get() == "Select Label":
        SyntaxAlignment_WarningMessage = tk.Label(SyntaxMainFrame, text="Warning: Must Select Label to Proceed", font=("Arial", 7))
        SyntaxAlignment_WarningMessage.grid(row=8, column=1)
    # RasterOutputDirectory = ""
    # for i in SyntaxDirectory.split("/")[:-1]:
    #     RasterOutputDirectory = RasterOutputDirectory+i+"/"
    else:
        if ChangeFigSize.get() == 1:
            X = int(X_Size.get())
            Y = int(Y_Size.get())
        else:
            X = 6
            Y = 6

        if SyntaxAlignmentVar.get() != "Auto":
            syntax_raster_df = syntax_data.make_syntax_raster(alignment_syllable=int(SyntaxAlignmentVar.get()))

        if SyntaxAlignmentVar.get() == "Auto":
            syntax_raster_df = syntax_data.make_syntax_raster()

        Bird_ID = SyntaxBirdID.get()
        FigTitle = str(Bird_ID)+" Syntax Raster"
        fig = avn.plotting.plot_syntax_raster(syntax_data, syntax_raster_df, title=FigTitle, figsize=(X, Y))

        if SyntaxAlignmentVar.get() == "Auto":
            fig.savefig(SyntaxOutputFolder+Bird_ID+"_Syntax_Rasterplot_Auto.png")
        else:
            fig.savefig(SyntaxOutputFolder+Bird_ID+"_Syntax_Rasterplot_"+str(SyntaxAlignmentVar.get())+".png")

        DisplayRaster = tk.Toplevel(gui)

        def SaveRaster_Function(fig):
            # fig.savefig(RasterOutputDirectory + FigTitle + ".png")
            Bird_ID = SyntaxBirdID.get()
            files = [('PNG Image', '*.png'), ('All Files', '*.*')]
            f = asksaveasfile(initialfile=str(Bird_ID) + "_RasterPlot_Syllable"+str(SyntaxAlignmentVar.get())+".png",
                              defaultextension=".png",
                              filetypes=[("All Files", "*.*"), ("PNG Image File", "*.png")])
            if len(str(f)) > 0:
                filename = str(f).split("\'")[1]
                try:
                    shutil.rmtree(filename)
                except:
                    pass
                fig.savefig(filename)

        SaveRaster = tk.Button(DisplayRaster, text="Save", command=lambda:SaveRaster_Function(fig))
        SaveRaster.grid(row=0, column=1)
        label_canvas = FigureCanvasTkAgg(fig, master=DisplayRaster)  # A tk.DrawingArea.
        label_canvas.draw()
        label_canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

Syntax_Raster_Input_Button = tk.Button(Syntax_RasterTab, text="Select Labeling File", command=lambda:FileExplorer("SyntaxRaster", "Input"))
Syntax_Raster_Input_Button.grid(row=0, column=0)
global Syntax_Raster_Input
Syntax_Raster_Input = tk.Label(Syntax_RasterTab, text=Dir_Width*" ", bg="light grey")
Syntax_Raster_Input.grid(row=0, column=1, sticky="w", padx=Padding_Width)
Syntax_Raster_BirdID_Label = tk.Label(Syntax_RasterTab, text="Bird ID: ").grid(row=1, column=0)
Syntax_Raster_BirdID = tk.Entry(Syntax_RasterTab, fg="grey",font=("Arial",15), justify="center", width=BirdID_Width)
Syntax_Raster_BirdID.insert(0, "Bird ID")
Syntax_Raster_BirdID.bind("<FocusIn>", focus_in)
Syntax_Raster_BirdID.bind("<FocusOut>", focus_out)
Syntax_Raster_BirdID.grid(row=1, column=1, sticky='w', padx=Padding_Width)
Syntax_Raster_SongLocation_Button = tk.Button(Syntax_RasterTab, text="Select Song Directory", command=lambda:FileExplorer("SyntaxRaster", "Song"))
Syntax_Raster_SongLocation_Button.grid(row=2, column=0)
global Syntax_Raster_SongLocation
Syntax_Raster_SongLocation = tk.Label(Syntax_RasterTab, text=Dir_Width*" ", bg="light grey")
Syntax_Raster_SongLocation.grid(row=2, column=1, sticky="w", padx=Padding_Width)
Syntax_Raster_Output_Button = tk.Button(Syntax_RasterTab, text="Select Output Directory", command=lambda:FileExplorer("SyntaxRaster", "Output"))
Syntax_Raster_Output_Button.grid(row=3, column=0)
global Syntax_Raster_Output
Syntax_Raster_Output = tk.Label(Syntax_RasterTab, text=Dir_Width*" ", bg="light grey")
Syntax_Raster_Output.grid(row=3, column=1, sticky="w", padx=Padding_Width)

global AlignmentChoices
global SyntaxAlignmentVar
global SyntaxAligment
SyntaxAlignmentVar = StringVar()
SyntaxAlignmentVar.set("Select Label")
AlignmentChoices = ["Select Label"]
SyntaxAlignment = tk.OptionMenu(Syntax_RasterTab, SyntaxAlignmentVar, *AlignmentChoices)
SyntaxAlignment.grid(row=4, column=1, sticky="w", padx=Padding_Width)
SyntaxAlignment_Label=tk.Label(Syntax_RasterTab, text="Select Alignment Variable: ").grid(row=4, column=0)

RasterButton = tk.Button(Syntax_RasterTab, text="Generate Raster Plot", command=lambda:GenerateRasterPlot())
RasterButton.grid(row=5, column=0, columnspan=2)

def Save_Syllable_Stats():
    global SyntaxDirectory
    Bird_ID = SyntaxBirdID.get()
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

    per_syll_stats.to_csv(OutputDir+Bird_ID+"_SyllableStats.csv")

# Syllable_Stats = tk.Button(SyntaxMainFrame, text="Save Syllable Stats", command=lambda:Save_Syllable_Stats())
# Syllable_Stats.grid(row=9, column=0)

global X_Size
X_Size = StringVar()
X_Size.set("10")
FixSize_X_Text = tk.Label(Syntax_HeatmapTab, text="X:").grid(row=0, column=2, sticky="e", padx=10)
FigSize_X = tk.Spinbox(Syntax_HeatmapTab, from_=1, to=20, state=DISABLED, textvariable=X_Size, width=15)
FigSize_X.grid(row=0, column=3, padx=10, sticky="w")

global Y_Size
Y_Size = StringVar()
Y_Size.set("10")
FixSize_Y_Text = tk.Label(Syntax_HeatmapTab, text="Y:").grid(row=1, column=2, sticky="e", padx=10)
FigSize_Y = tk.Spinbox(Syntax_HeatmapTab, from_=1, to=20, state=DISABLED, textvariable=Y_Size, width=15)
FigSize_Y.grid(row=1, column=3, padx=10, sticky="w")

def ChangeFigureSize():
    if ChangeFigSize.get() == 1:
        FigSize_X.config(state=NORMAL)
        FigSize_Y.config(state=NORMAL)
    if ChangeFigSize.get() == 0:
        FigSize_X.config(state=DISABLED)
        FigSize_Y.config(state=DISABLED)

global ChangeFigSize
ChangeFigSize = IntVar()
CustomFigSize = tk.Checkbutton(Syntax_HeatmapTab, text="Custom Figure Size", variable=ChangeFigSize, command=lambda:ChangeFigureSize())
CustomFigSize.grid(row=2, column=2, columnspan=2, padx=10)

### Plain Spectrogram Generation - Whole File ###
PlainSpectrograms = tk.Frame(gui, width=MasterFrameWidth, height=MasterFrameHeight)
# notebook.add(PlainSpectrograms, text="Plain Spectrograms")

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
PlainOutputFolder_Button = tk.Button(Plain_Folder, text="Select Output Directory",
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
PlainOutputAlt_Button = tk.Button(PlainSpectroAlt, text="Select Output Directory", command=lambda:FileExplorer("Plain_Files", "Output"))
PlainOutputAlt_Button.grid(row=2, column = 0)

global PlainOutputAlt_Label
PlainOutputAlt_Label = tk.Label(PlainSpectroAlt, text = Dir_Width*" ", bg="light grey")
PlainOutputAlt_Label.grid(row=2, column=1, padx=Padding_Width)
PlainSpectroRunAlt = tk.Button(PlainSpectroAlt, text="Create Blank Spectrograms",
                            command=lambda: PrintPlainSpectrogramsAlt())
PlainSpectroRunAlt.grid(row=3, column=0, columnspan=2)
PlainSettingsFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
# PlainNotebook.add(PlainSettingsFrame, text="Advanced Settings")
# PlainInfoFrame = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
# PlainNotebook.add(PlainInfoFrame, text="Info")

# Plain Settings -- Uses same parameters as labeling #
def ResetPlainSetting(Variable, EntryList, ErrorLabel):
    DefaultValues = [500,15000,.00005,20,-28,512,512, 128,300]
    global PlainErrorLabels
    if Variable == "all":
        i = -1
        for var in EntryList:
            i+=1
            var.delete(0, END)
            var.insert(0, str(DefaultValues[i]))
            var.config(bg="white")
            var.update()
        for label in PlainErrorLabels:
            label.config(text="")

    else:
        Variable.delete(0, END)
        Variable.insert(0,str(DefaultValues[EntryList.index(Variable)]))
        Variable.config(bg="white")
        Variable.update()
        ErrorLabel.config(text="")

bandpass_lower_cutoff_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
bandpass_lower_cutoff_Plain_Error.grid(row=1, column=1)

PlainSettingsDialogTitle = tk.Label(PlainSettingsFrame, text="", justify="center", font=("Arial", 25, "bold"), height=Dialog_TitleHeight, width=Dialog_TitleWidth)
PlainSettingsDialogTitle.grid(row=0, column=4)
PlainSettingsDialog = tk.Label(PlainSettingsFrame, text="", justify="center")
PlainSettingsDialog.grid(row=1, column=4, rowspan=8)

bandpass_lower_cutoff_text_Plain = tk.Label(PlainSettingsFrame, text="bandpass_lower_cutoff").grid(row=0, column=0)
bandpass_lower_cutoff_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
bandpass_lower_cutoff_entry_Plain.insert(0, "500")
bandpass_lower_cutoff_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(bandpass_lower_cutoff_entry_Plain, "bandpass_lower_cutoff_entry_Plain",bandpass_lower_cutoff_Plain_Error))
bandpass_lower_cutoff_entry_Plain.grid(row=0, column=1)
bandpass_lower_cutoff_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(bandpass_lower_cutoff_entry_Plain, LabelSpecList,bandpass_lower_cutoff_Plain_Error)).grid(row=0, column=2)
bandpass_lower_cutoff_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
bandpass_lower_cutoff_moreinfo_Plain.grid(row=0, column=3, sticky=W)
bandpass_lower_cutoff_moreinfo_Plain.bind("<Enter>", MoreInfo)
bandpass_lower_cutoff_moreinfo_Plain.bind("<Leave>", LessInfo)

bandpass_upper_cutoff_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
bandpass_upper_cutoff_Plain_Error.grid(row=3, column=1)
bandpass_upper_cutoff_text_Plain = tk.Label(PlainSettingsFrame, text="bandpass_upper_cutoff").grid(row=2,column=0)
bandpass_upper_cutoff_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
bandpass_upper_cutoff_entry_Plain.insert(0, "15000")
bandpass_upper_cutoff_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(bandpass_upper_cutoff_entry_Plain,"bandpass_upper_cutoff_entry_Plain",bandpass_upper_cutoff_Plain_Error))
bandpass_upper_cutoff_entry_Plain.grid(row=2, column=1)
bandpass_upper_cutoff_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(bandpass_upper_cutoff_entry_Plain, LabelSpecList,bandpass_upper_cutoff_Plain_Error)).grid(row=2, column=2)
bandpass_upper_cutoff_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
bandpass_upper_cutoff_moreinfo_Plain.grid(row=2, column=3, sticky=W)
bandpass_upper_cutoff_moreinfo_Plain.bind("<Enter>", MoreInfo)
bandpass_upper_cutoff_moreinfo_Plain.bind("<Leave>", LessInfo)

a_min_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
a_min_Plain_Error.grid(row=5, column=1)
a_min_text_Plain = tk.Label(PlainSettingsFrame, text="a_min").grid(row=4,column=0)
a_min_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
a_min_entry_Plain.insert(0, "0.00001")
a_min_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(a_min_entry_Plain,"a_min_entry_Plain",a_min_Plain_Error))
a_min_entry_Plain.grid(row=4, column=1)
a_min_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(a_min_entry_Plain, LabelSpecList,a_min_Plain_Error)).grid(row=4, column=2)
a_min_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
a_min_moreinfo_Plain.grid(row=4, column=3, sticky=W)
a_min_moreinfo_Plain.bind("<Enter>", MoreInfo)
a_min_moreinfo_Plain.bind("<Leave>", LessInfo)

ref_db_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
ref_db_Plain_Error.grid(row=7, column=1)
ref_db_text_Plain = tk.Label(PlainSettingsFrame, text="ref_db").grid(row=6,column=0)
ref_db_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
ref_db_entry_Plain.insert(0, "20")
ref_db_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(ref_db_entry_Plain,"ref_db_entry_Plain",ref_db_Plain_Error))
ref_db_entry_Plain.grid(row=6, column=1)
ref_db_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(ref_db_entry_Plain, LabelSpecList,ref_db_Plain_Error)).grid(row=6, column=2)
ref_db_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
ref_db_moreinfo_Plain.grid(row=6, column=3, sticky=W)
ref_db_moreinfo_Plain.bind("<Enter>", MoreInfo)
ref_db_moreinfo_Plain.bind("<Leave>", LessInfo)

min_level_db_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
min_level_db_Plain_Error.grid(row=9, column=1)
min_level_db_text_Plain = tk.Label(PlainSettingsFrame, text="min_level_db").grid(row=8,column=0)
min_level_db_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
min_level_db_entry_Plain.insert(0, "-28")
min_level_db_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(min_level_db_entry_Plain,'min_level_db_entry_Plain',min_level_db_Plain_Error))
min_level_db_entry_Plain.grid(row=8, column=1)
min_level_db_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(min_level_db_entry_Plain, LabelSpecList,min_level_db_Plain_Error)).grid(row=8, column=2)
min_level_db_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
min_level_db_moreinfo_Plain.grid(row=8, column=3, sticky=W)
min_level_db_moreinfo_Plain.bind("<Enter>", MoreInfo)
min_level_db_moreinfo_Plain.bind("<Leave>", LessInfo)

n_fft_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
n_fft_Plain_Error.grid(row=11, column=1)
n_fft_text_Plain = tk.Label(PlainSettingsFrame, text="n_fft").grid(row=10,column=0)
n_fft_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
n_fft_entry_Plain.insert(0, "512")
n_fft_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(n_fft_entry_Plain,'n_fft_entry_Plain',n_fft_Plain_Error))
n_fft_entry_Plain.grid(row=10, column=1)
n_fft_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(n_fft_entry_Plain, LabelSpecList,n_fft_Plain_Error)).grid(row=10, column=2)
n_fft_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
n_fft_moreinfo_Plain.grid(row=10, column=3, sticky=W)
n_fft_moreinfo_Plain.bind("<Enter>", MoreInfo)
n_fft_moreinfo_Plain.bind("<Leave>", LessInfo)

win_length_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
win_length_Plain_Error.grid(row=13, column=1)
win_length_text_Plain = tk.Label(PlainSettingsFrame, text="win_length").grid(row=12,column=0)
win_length_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
win_length_entry_Plain.insert(0, "512")
win_length_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(win_length_entry_Plain,'win_length_entry_Plain',win_length_Plain_Error))
win_length_entry_Plain.grid(row=12, column=1)
win_length_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(win_length_entry_Plain, LabelSpecList,win_length_Plain_Error)).grid(row=12, column=2)
win_length_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
win_length_moreinfo_Plain.grid(row=12, column=3, sticky=W)
win_length_moreinfo_Plain.bind("<Enter>", MoreInfo)
win_length_moreinfo_Plain.bind("<Leave>", LessInfo)

hop_length_Plain_Error = tk.Label(PlainSettingsFrame, text="", font=ErrorFont)
hop_length_Plain_Error.grid(row=15, column=1)
hop_length_text_Plain = tk.Label(PlainSettingsFrame, text="hop_length").grid(row=14,column=0)
hop_length_entry_Plain = tk.Entry(PlainSettingsFrame, justify="center")
hop_length_entry_Plain.insert(0, "128")
hop_length_entry_Plain.bind("<FocusOut>", lambda value:Validate_Settings(hop_length_entry_Plain,'hop_length_entry_Plain',hop_length_Plain_Error))
hop_length_entry_Plain.grid(row=14, column=1)
hop_length_reset_Plain = tk.Button(PlainSettingsFrame, text="Reset",command=lambda: ResetPlainSetting(hop_length_entry_Plain, LabelSpecList,hop_length_Plain_Error)).grid(row=14, column=2)
hop_length_moreinfo_Plain = tk.Button(PlainSettingsFrame, text='?', state="disabled", fg="black")
hop_length_moreinfo_Plain.grid(row=14, column=3, sticky=W)
hop_length_moreinfo_Plain.bind("<Enter>", MoreInfo)
hop_length_moreinfo_Plain.bind("<Leave>", LessInfo)

### Timing Module ###
TimingTab = tk.Frame(gui, width=MasterFrameWidth, height=MasterFrameHeight)
notebook.add(TimingTab, text="Timing")
TimingNotebook = ttk.Notebook(TimingTab)
TimingNotebook.grid(row=1)
# TimingMainFrame = tk.Frame(TimingTab, width=MasterFrameWidth, height=MasterFrameHeight)
# TimingNotebook.add(TimingMainFrame, text="Home")
SyllableTiming = tk.Frame(TimingTab, width=MasterFrameWidth, height=MasterFrameHeight)
TimingNotebook.add(SyllableTiming, text="Syllable Timing")
RhythmSpectrogram = tk.Frame(TimingTab, width=MasterFrameWidth, height=MasterFrameHeight)
TimingNotebook.add(RhythmSpectrogram, text="Rhythm Spectrogram")
TimingSettings = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
TimingNotebook.add(TimingSettings, text="Advanced Settings")

# Syllable Timing #
TimingInput_Button = tk.Button(SyllableTiming, text="Select Segmentation File", command=lambda:FileExplorer("Timing", "Input"))
TimingInput_Button.grid(row=0, column=0)
global TimingInput_Text
TimingInput_Text = tk.Label(SyllableTiming, text=Dir_Width*" ", bg="light grey")
TimingInput_Text.grid(row=0, column=1, columnspan=2, padx=Padding_Width, sticky="W")

TimingSongPath_Button = tk.Button(SyllableTiming, text="Select Song Directory", command=lambda:FileExplorer("Timing", "Songs"))
TimingSongPath_Button.grid(row=1, column=0)
global TimingSongPath
TimingSongPath = tk.Label(SyllableTiming, text=Dir_Width*" ", bg="light grey")
TimingSongPath.grid(row=1, column=1, sticky="w", padx=Padding_Width)

TimingOutput_Button = tk.Button(SyllableTiming, text="Select Output Directory", command=lambda:FileExplorer("Timing", "Output"))
TimingOutput_Button.grid(row=3, column=0)
global TimingOutput_Text
TimingOutput_Text = tk.Label(SyllableTiming, text=Dir_Width * " ", bg="light grey")
TimingOutput_Text.grid(row=3, column=1, columnspan=2, padx=Padding_Width, sticky="W")
Timing_BirdID_Label = tk.Label(SyllableTiming, text="Bird ID: ").grid(row=2, column=0)
global Timing_BirdID
Timing_BirdID = tk.Entry(SyllableTiming, font=("Arial", 15), justify="center", fg="grey",
                               width=BirdID_Width)
Timing_BirdID.insert(0, "Bird ID")
Timing_BirdID.bind("<FocusIn>", focus_in)
Timing_BirdID.bind("<FocusOut>", focus_out)
Timing_BirdID.grid(row=2, column=1, pady=5, sticky="W", padx=Padding_Width)

RunTiming = tk.Button(SyllableTiming, text="Run Timing", command=lambda:Timing())
RunTiming.grid(row=4, column=0, columnspan=3)

# Rhythm Spectrogram #
RhythmSpectrogram_InputButton = tk.Button(RhythmSpectrogram, text="Select Song Directory", command=lambda:FileExplorer("Timing_Rhythm", "Input"))
RhythmSpectrogram_InputButton.grid(row=0, column=0, sticky="w", padx=Padding_Width)
RhythmSpectrogram_Input = tk.Label(RhythmSpectrogram, text=Dir_Width * " ", bg="light grey")
RhythmSpectrogram_Input.grid(row=0, column=1, sticky="w")
RhythmSpectrogram_BirdID_Label = tk.Label(RhythmSpectrogram, text="Bird ID: ").grid(row=1, column=0)
global RhythmSpectrogram_BirdID
RhythmSpectrogram_BirdID = tk.Entry(RhythmSpectrogram,font=("Arial", 15), justify="center", fg="grey",
                               width=BirdID_Width)
RhythmSpectrogram_BirdID.insert(0, "Bird ID")
RhythmSpectrogram_BirdID.bind("<FocusIn>", focus_in)
RhythmSpectrogram_BirdID.bind("<FocusOut>", focus_out)
RhythmSpectrogram_BirdID.grid(row=1, column=1, pady=5, sticky="W", padx=Padding_Width)
RhythmSpectrogram_OutputButton = tk.Button(RhythmSpectrogram, text="Select Output Directory", command=lambda:FileExplorer("Timing_Rhythm", "Output"))
RhythmSpectrogram_OutputButton.grid(row=2, column=0, sticky="w", padx=Padding_Width)
RhythmSpectrogram_Output = tk.Label(RhythmSpectrogram, text=Dir_Width * " ", bg="light grey")
RhythmSpectrogram_Output.grid(row=2, column=1, sticky="w")
RhythmSpectrogram_RunButton = tk.Button(RhythmSpectrogram, text="Run", command=lambda:TimingRhythm())
RhythmSpectrogram_RunButton.grid(row=4, column=0, columnspan=2)

# Timing Settings #
max_gap_Label = tk.Label(TimingSettings, text="max_gap", justify="center", font=ErrorFont).grid(row=0, column=0)
max_gap_Entry = tk.Entry(TimingSettings)
max_gap_Entry.insert(0, "0.2")
max_gap_Entry.grid(row=0, column=1, sticky="W", padx=Padding_Width)

def MaxGapReset():
    max_gap_Entry.delete(0,END)
    max_gap_Entry.insert(0,"0.2")
    max_gap_Entry.update()
max_gap_reset = tk.Button(TimingSettings, text="Reset", command=lambda:MaxGapReset()).grid(row=0, column=2)

def MaxGapLoad():
    MetadataFile = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
    if len(MetadataFile) > 0:
        Metadata = pd.read_csv(MetadataFile)
        max_gap_Entry.delete(0, END)
        max_gap_Entry.insert(0, Metadata["max_gap"][0])
        max_gap_Entry.update()
max_gap_load = tk.Button(TimingSettings, text="Load Timing Setting", command=lambda:MaxGapLoad()).grid(row=0, column=3)

smoothing_window_Label= tk.Label(TimingSettings, text="smoothing_window", justify="center", font=ErrorFont).grid(row=1, column=0)
smoothing_window_Entry = tk.Entry(TimingSettings)
smoothing_window_Entry.insert(0, "1")
smoothing_window_Entry.grid(row=1, column=1, sticky="W", padx=Padding_Width)

def SmoothingWindowReset():
    smoothing_window_Entry.delete(0,END)
    smoothing_window_Entry.insert(0,"1")
    smoothing_window_Entry.update()
smoothing_window_reset = tk.Button(TimingSettings, text="Reset", command=lambda:SmoothingWindowReset()).grid(row=1, column=2)

def SmoothingWindowLoad():
    MetadataFile = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
    if len(MetadataFile) > 0:
        Metadata = pd.read_csv(MetadataFile)
        smoothing_window_Entry.delete(0, END)
        smoothing_window_Entry.insert(0, Metadata["smoothing_window"][0])
        smoothing_window_Entry.update()
smoothing_window_load = tk.Button(TimingSettings, text="Load Rhythm Setting", command=lambda:SmoothingWindowLoad()).grid(row=1, column=3)

def ResetTimingSettings():
    max_gap_Entry.delete(0,END)
    max_gap_Entry.insert(0, "0.2")
    max_gap_Entry.update()

    smoothing_window_Entry.delete(0, END)
    smoothing_window_Entry.insert(0, "1")
    smoothing_window_Entry.update()

def LoadTimingSettings():
    SettingsMetadata = filedialog.askopenfilename(filetypes=[(".csv files", "*.csv")])
    if SettingsMetadata != "":
        SettingsMetadata_df = pd.read_csv(SettingsMetadata)

        max_gap_Entry.delete(0, END)
        max_gap_Entry.insert(0, str(SettingsMetadata_df["max_gap"][0]))
        max_gap_Entry.update()

        smoothing_window_Entry.delete(0, END)
        smoothing_window_Entry.insert(0, str(SettingsMetadata_df["smoothing_window"][0]))
        smoothing_window_Entry.update()

Reset_Timing_Settings = tk.Button(TimingSettings, text="Reset Settings", command=lambda:ResetTimingSettings())
Reset_Timing_Settings.grid(row=3, column=1)

# Load_Timing_Settings = tk.Button(TimingSettings, text="Load Settings", command=lambda:LoadTimingSettings())
# Load_Timing_Settings.grid(row=3, column=0)

# Run Everything Tab #
RunAllTab = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
notebook.add(RunAllTab, text="Run All")
RunAll_Title = tk.Label(RunAllTab, text="This will calculate the complete AVN feature set \n"
                                        "for all labeled songs in the selected label table. \n \n"
                                        "Settings and parameters for these features can be adjusted \n"
                                        "within each feature's dedicated tab", font=("Arial",15)).grid(row=0, columnspan=2)
RunAllPadding = tk.Label(RunAllTab, text="", font=("Arial",20)).grid(row=1)

RunAll_ImportantFrame = tk.Frame(RunAllTab, width=400, height=400, highlightbackground="black", highlightthickness=1)
RunAll_ImportantFrame.grid(row=2, column=0, columnspan=2)
RunAll_Input = tk.Button(RunAll_ImportantFrame, text="Select Labeling File:", command=lambda:FileExplorer("RunAll","Input"))
RunAll_Input.grid(row=1, column=0)
global RunAll_InputFileDisplay
RunAll_InputFileDisplay = tk.Label(RunAll_ImportantFrame, text=Dir_Width*" ", bg="light grey")
RunAll_InputFileDisplay.grid(row=1, column=1)

RunAll_SongPath_Button = tk.Button(RunAll_ImportantFrame, text="Select Song Directory", command=lambda:FileExplorer("RunAll","Songs"))
RunAll_SongPath_Button.grid(row=2, column=0)
global RunAll_SongPath
RunAll_SongPath=tk.Label(RunAll_ImportantFrame, text=Dir_Width*" ", bg="light grey")
RunAll_SongPath.grid(row=2, column=1)

RunAll_BirdID_Label = tk.Label(RunAll_ImportantFrame, text="Bird ID:").grid(row=3, column=0)
RunAll_BirdID_Entry = tk.Entry(RunAll_ImportantFrame, font=("Arial", 15), justify="center", fg="grey", bg="white",
                               width=BirdID_Width)
RunAll_BirdID_Entry.grid(row=3, column=1)
RunAll_BirdID_Entry.insert(0, "Bird ID")
RunAll_BirdID_Entry.bind("<FocusIn>", focus_in)
RunAll_BirdID_Entry.bind("<FocusOut>", focus_out)

RunAll_Output = tk.Button(RunAll_ImportantFrame, text="Select Output Folder:",command=lambda:FileExplorer("RunAll","Output"))
RunAll_Output.grid(row=4, column=0)
global RunAll_OutputFileDisplay
RunAll_OutputFileDisplay = tk.Label(RunAll_ImportantFrame, text=Dir_Width*" ", bg="light grey")
RunAll_OutputFileDisplay.grid(row=4, column=1)

RunAllPadding2 = tk.Label(RunAllTab, text="", font=("Arial",20)).grid(row=5)
global RunAll_StartRow
RunAll_StartRow = 6

RunAll_RunButton = tk.Button(RunAllTab, text="Run", command=lambda:RunAllModules())
RunAll_RunButton.grid(row=RunAll_StartRow+8, column=0, columnspan=2)

### Similarity Scoring ###
SimilarityTab = tk.Frame(gui, width=MasterFrameWidth, height=MasterFrameHeight)
notebook.add(SimilarityTab, text="Similarity Scoring")
SimilarityNotebook = ttk.Notebook(SimilarityTab)
SimilarityNotebook.grid()
PrepSpectrograms = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SimilarityNotebook.add(PrepSpectrograms, text = "Prep Spectrograms")
OutputSimilarity = tk.Frame(width=MasterFrameWidth, height=MasterFrameHeight)
SimilarityNotebook.add(OutputSimilarity, text="Compare Birds")

# Prep Spectrograms Tab #
SimilarityInput_Button = tk.Button(PrepSpectrograms, text="Select Segmentation File", command=lambda:FileExplorer("Similarity_Prep", "Input"))
SimilarityInput_Button.grid(row=0, column=0)
global SimilarityInput
SimilarityInput = tk.Label(PrepSpectrograms, text=Dir_Width*" ", bg="light grey")
SimilarityInput.grid(row=0, column=1, columnspan=2, sticky="W", padx=Padding_Width)

SimilaritySongPath_Button = tk.Button(PrepSpectrograms, text="Select Song Directory", command=lambda:FileExplorer("Similarity_Prep", "Songs"))
SimilaritySongPath_Button.grid(row=1, column=0)
global SimilaritySongPath
SimilaritySongPath = tk.Label(PrepSpectrograms, text=Dir_Width*" ", bg="light grey")
SimilaritySongPath.grid(row=1, column=1, sticky="w", padx=Padding_Width)

Similarity_BirdID_Label = tk.Label(PrepSpectrograms, text="Bird ID: ").grid(row=3, column=0)
global Similarity_BirdID
Similarity_BirdID = tk.Entry(PrepSpectrograms,font=("Arial", 15), justify="center", fg="grey", bg="white",
                               width=BirdID_Width)
Similarity_BirdID.grid(row=3, column=1, sticky="W", padx=Padding_Width)
Similarity_BirdID.insert(0, "Bird ID")
Similarity_BirdID.bind("<FocusIn>", focus_in)
Similarity_BirdID.bind("<FocusOut>", focus_out)

SimilarityOutput_Button = tk.Button(PrepSpectrograms, text="Select Output Directory", command=lambda:FileExplorer("Similarity_Prep", "Output"))
SimilarityOutput_Button.grid(row=4, column=0)
global SimilarityOutput
SimilarityOutput = tk.Label(PrepSpectrograms, text=Dir_Width*" ", bg="light grey")
SimilarityOutput.grid(row=4, column=1, columnspan=2, sticky="W", padx=Padding_Width)

RunSimilarity_Button = tk.Button(PrepSpectrograms, text="Run", command=lambda:SimilarityScoring_Prep())
RunSimilarity_Button.grid(row=5, column=0, columnspan=2)

# Create EMD and UMAP #
Title_Left = tk.Label(OutputSimilarity, text="Bird 1", justify="center").grid(row=0, column=0, columnspan=2, sticky="N")
Title_Right = tk.Label(OutputSimilarity, text="Bird 2", justify="center").grid(row=0, column=4, columnspan=2, sticky="N")

SimilarityInput_Button2 = tk.Button(OutputSimilarity, text="Select Spectrograms",
                                    command=lambda: FileExplorer("Similarity_Out1", "Input"))
SimilarityInput_Button2.grid(row=1, column=0)
global SimilarityInput2
SimilarityInput2 = tk.Label(OutputSimilarity, text=Dir_Width * " ", bg="light grey")
SimilarityInput2.grid(row=1, column=1, columnspan=2, sticky="W")
Similarity_BirdID_Label2 = tk.Label(OutputSimilarity, text="Bird ID: ").grid(row=2, column=0)
global Similarity_BirdID2
Similarity_BirdID2 = tk.Entry(OutputSimilarity, font=("Arial", 15), justify="center", fg="grey", bg="white",
                              width=BirdID_Width)
Similarity_BirdID2.grid(row=2, column=1, padx=Padding_Width, sticky="w")
Similarity_BirdID2.insert(0, "Bird ID")
Similarity_BirdID2.bind("<FocusIn>", focus_in)
Similarity_BirdID2.bind("<FocusOut>", focus_out)

PaddingRow = tk.Label(OutputSimilarity, text=" ", font=("Arial", 15)).grid(row=3)
SimilarityOutput_Button2 = tk.Button(OutputSimilarity, text="Select Output Directory",
                                     command=lambda: FileExplorer("Similarity_Out1", "Output"),justify="center")
SimilarityOutput_Button2.grid(row=4, column=1, columnspan=2)
global SimilarityOutput2
SimilarityOutput2 = tk.Label(OutputSimilarity, text=Dir_Width * " ", bg="light grey")
SimilarityOutput2.grid(row=4, column=3, columnspan=2, sticky="W", padx=Padding_Width)

# Create Similarity EMD for second bird #
PaddingColumn = tk.Label(OutputSimilarity, text=" "*5).grid(row=0, column=3)
SimilarityInput_Button3 = tk.Button(OutputSimilarity, text="Select Spectrograms",
                                   command=lambda: FileExplorer("Similarity_Out2", "Input"))
SimilarityInput_Button3.grid(row=1, column=4)
global SimilarityInput3
SimilarityInput3 = tk.Label(OutputSimilarity, text=Dir_Width * " ", bg="light grey")
SimilarityInput3.grid(row=1, column=5, columnspan=2, sticky="W", padx=Padding_Width)
Similarity_BirdID_Label3 = tk.Label(OutputSimilarity, text="Bird ID: ").grid(row=2, column=4)
global Similarity_BirdID3
Similarity_BirdID3 = tk.Entry(OutputSimilarity, font=("Arial", 15), justify="center", fg="grey", bg="white",
                             width=BirdID_Width)
Similarity_BirdID3.grid(row=2, column=5, sticky="W", padx=Padding_Width)
Similarity_BirdID3.insert(0, "Bird ID")
Similarity_BirdID3.bind("<FocusIn>", focus_in)
Similarity_BirdID3.bind("<FocusOut>", focus_out)

PadRunButton = tk.Label(OutputSimilarity, text=" ", font=("Arial", 15)).grid(row=5)

global SaveEmbedding
SaveEmbedding = IntVar()
SaveEmbedding_Checkbox = tk.Checkbutton(OutputSimilarity, text="Save Embedding Data", variable=SaveEmbedding)
SaveEmbedding_Checkbox.grid(row=6, column=1, columnspan=4, sticky="S")
RunSimilarity_Comparison = tk.Button(OutputSimilarity, text="Compare Birds", command=lambda: SimilarityScoring_Output())
RunSimilarity_Comparison.grid(row=7, column=1, columnspan=4, sticky="S")

notebook.add(PlainSpectrograms, text="Plain Spectrograms")

ttk.Style().theme_use("clam")
gui.mainloop()