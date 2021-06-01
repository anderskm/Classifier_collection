import src.data.datasets.DS_OSD_base as superDataset
import os
import random
from operator import itemgetter
# import numpy as np
# import PIL
# import tensorflow as tf
# import os
# import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, white clover repetition',
                    rawFolder = 'data/raw/extracted_repeat__masked/Randers/Hvidkloever',
                    processFolder = 'data/processed/OSD_wc_repeat',
                    numShards = 10):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
