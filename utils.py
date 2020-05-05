import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import csv

categories = ['Ballad', 'Rock', 'Pop', 'Hiphop', 'Trot', 'Dance', 'RnB']
mapper = pd.ExcelFile('~/Downloads/mapper.xlsx')

def npyLengthSaver():
    musicFullName2NpyLength = {}
    for category in categories:
        path = f"/Users/piljae/Desktop/SeparatedNpys/{category}"
        npys = os.listdir(path)
        for npy in tqdm(npys):
            fullpath = os.path.join(path, npy)
            basepath = os.path.basename(fullpath)
            filename = os.path.splitext(basepath)[0]
            npyLength = len(np.load(fullpath))
            musicFullName2NpyLength[filename] = npyLength
    with open('/Users/piljae/VoiceClassifier/music2npylength.pickle', 'wb') as f:
        pickle.dump(musicFullName2NpyLength, f, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    npyLengthSaver()