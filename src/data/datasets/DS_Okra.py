import src.data.datasets.DS_Seeds as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Okra seed dataset',
                    rawFolder = 'data/raw/Okra',
                    processFolder = 'data/processed/Okra',
                    numShards = 10):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _filename_to_class(self, filename):
        # Extract class names from filename
        classes = self._filename_to_classes(filename)
        # Grab "germination" class of for this image in the image sequence
        class_out = [classes[0][0]]
        return class_out

    def _filename_to_classes(self, filename):
        # Returns all class combinations available from filename (germ, normal for current, next and last)
        basename_without_extension = os.path.splitext(os.path.basename(filename))[0]
        basename_parts = basename_without_extension.split('_')
        class_this = self._class_idx_to_class_name(int(basename_parts[9])) # Class for this image in image sequence
        class_next = self._class_idx_to_class_name(int(basename_parts[10])) # Class for this image in image sequence
        class_end = self._class_idx_to_class_name(int(basename_parts[11])) # Class at end of image sequence
        return [class_this, class_next, class_end]

    def _class_idx_to_class_name(self, class_idx):
        if (class_idx < 4):
            class_abnorm = 'Normal'
        else:
            class_abnorm = 'Abnormal'

        if (class_idx == 1) or (class_idx == 4):
            class_germ = 'Not germinated'
        else:
            class_germ = 'Germinated'
        
        return [class_germ, class_abnorm]
