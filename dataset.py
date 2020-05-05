from glob import glob
import os
import random
import pickle

from torch.utils import data
import pandas as pd
import numpy as np
import librosa
import torch

'''
    Naming Conventions
    
    musicFullName 은 /Path/musicFullName.npy 에서의 musicFullName 을 말함
    OfficialMusicName 은 mapper 에서의 music 을 말함
'''

# TODO : Split the dataset into Train and Test

class VoiceDataset(data.Dataset):
    def __init__(self, sampleNumber=100):
        # Librosa related global vars
        self.sr = 22050
        self.n_fft = 2048
        self.hop_length = 512

        self.sampleNumber = sampleNumber
        self.sampleLength = 200000 # The number of npy samples to use as input -> 9.07 sec

        # Others
        self.mapper = pd.ExcelFile('~/Downloads/mapper.xlsx')
        self.categories = ['Ballad', 'Rock', 'Pop', 'Hiphop', 'Trot', 'Dance', 'RnB']
        self.musicFullName2FullPath = {}
        self.musicFullName2Artist = {}
        self.musicFullName2OfficialMusicName = {}
        self.musicFullName2Tags = {}
        self.musicFullName2Category = {}
        with open('./music2npylength.pickle', 'rb') as f:
            self.musicFullName2SampleNumber = pickle.load(f)
        self.dictInitializer()
        self.musicFullNames = list(self.musicFullName2FullPath.keys())
        self.npyMusicFullNameStartIdxEndIdxPairs = [] # [[musicFullName, startIdx, endIdx], ... ]
        self.pairInitializer()

    def dictInitializer(self):
        for category in self.categories:
            DF = pd.read_excel(self.mapper, category)
            path = f"/Users/piljae/Desktop/SeparatedNpys/{category}"
            npys = os.listdir(path)
            for npy in npys:
                fullpath = os.path.join(path, npy)
                basepath = os.path.basename(fullpath)
                filename = os.path.splitext(basepath)[0]
                filteredDF = DF.loc[DF['file'].map(lambda x: x.replace(" ", "")) == filename]
                assert len(filteredDF) == 1, "Should return only a single row, no duplicates"
                artist = filteredDF.iloc[0, 1]
                officialMusicName = filteredDF.iloc[0, 2]
                _tags = filteredDF.iloc[0, 3:].values
                tags = [tag for tag in _tags if pd.notnull(tag)]
                self.musicFullName2FullPath[filename] = fullpath
                self.musicFullName2Artist[filename] = artist
                self.musicFullName2OfficialMusicName[filename] = officialMusicName
                self.musicFullName2Tags[filename] = tags
                self.musicFullName2Category[filename] = category

    def __len__(self):
        return len(self.npyMusicFullNameStartIdxEndIdxPairs)

    def __getitem__(self, index):
        pair = self.npyMusicFullNameStartIdxEndIdxPairs[index]
        musicFullName, startSample, endSample = pair
        npyPath = self.musicFullName2FullPath[musicFullName]
        npy = np.load(npyPath)
        partialNpy = npy[startSample:endSample]

        # Feature
        partialMelSpectrogram = self.npy2melspectrogram(partialNpy)
        
        # Labels
        category = self.musicFullName2Category[musicFullName]
        tags = self.musicFullName2Tags[musicFullName]

        return partialMelSpectrogram, category, tags

    def pairInitializer(self):
        for musicFullName in self.musicFullNames:
            self.randomSampleFromSingleNpy(musicFullName)

    def randomSampleFromSingleNpy(self, musicFullName):
        """
            Randomly select the starting point between intervals of 1/6 to 5/6
            Returns a list of [start idx, end idx]
        """
        sampleNumber = self.musicFullName2SampleNumber[musicFullName]
        startingSample = int(sampleNumber * (1/6))
        endingSample = int(sampleNumber * (5/6))
        randomStartingSamples = random.sample(range(startingSample, endingSample), self.sampleNumber)
        for startSample in randomStartingSamples:
            endSample = startSample + self.sampleLength
            self.npyMusicFullNameStartIdxEndIdxPairs.append([musicFullName, startSample, endSample])

    def npy2melspectrogram(self, npy):
        melspec = librosa.feature.melspectrogram(npy, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        return melspec

    def npy2duration(self, npy):
        duration = librosa.get_duration(npy)
        return duration

def test(number):
    random_sample = np.random.randn(number)
    print(len(random_sample))
    dur = librosa.get_duration(random_sample)
    print(f"{number} samples equal to {dur} seconds.")

if __name__ == '__main__':
    # test(200000)
    dataset = VoiceDataset()
