import src.data.datasets.DS_Okra_D0 as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Barley seed dataset, D0',
                    rawFolder = 'data/raw/Barley',
                    processFolder = 'data/processed/Barley_D0',
                    numShards = 1):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
