import src.data.datasets.DS_OSD as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, Weeds',
                    rawFolder = 'data/raw/OSD_Weeds',
                    processFolder = 'data/processed/OSD_Weeds',
                    numShards = 1):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
