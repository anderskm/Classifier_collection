import src.data.datasets.DS_OSD_base as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, Weeds, seed only, new focus',
                    rawFolder = 'data/raw/extracted_weeds_new_focus__masked',
                    processFolder = 'data/processed/OSD_Weeds_seed_new_focus',
                    numShards = 1):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
