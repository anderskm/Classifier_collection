import src.data.datasets.DS_Okra_Next as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Barley seed dataset, next',
                    rawFolder = 'data/raw/Barley',
                    processFolder = 'data/processed/Barley_Next',
                    numShards = 4):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
