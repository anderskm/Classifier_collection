import src.data.datasets.DS_OSD_base as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, Weeds, seed only, new white clover focus',
                    rawFolder = 'data/raw/OSD_Weeds_seed_new_wc',
                    processFolder = 'data/processed/OSD_Weeds_seed_new_wc',
                    numShards = 1):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
