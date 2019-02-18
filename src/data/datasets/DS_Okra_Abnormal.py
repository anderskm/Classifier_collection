import src.data.datasets.DS_Okra as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Okra seed dataset, abnormal',
                    rawFolder = 'data/raw/Okra',
                    processFolder = 'data/processed/Okra_Abnormal',
                    numShards = 10):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _filename_to_class(self, filename):
        # Extract class names from filename
        classes = self._filename_to_classes(filename)
        # Grab "abnormal" class of the for this image in the image sequence
        class_out = [classes[0][1]]
        return class_out