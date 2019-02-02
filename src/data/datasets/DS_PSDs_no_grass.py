import src.data.datasets.DS_PSDs as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

import sys
from six.moves import urllib
import zipfile
import shutil

    # TODO Pre-processing factory
    # random rotation
    # - output size: same or valid
    # -- same: keep same dimensios as input
    # -- valid: crop to 1/sqrt(2) of input
    # Normalize:
    # - Subtract mean
    # - divide by standard deviation
    # -- Fixed, from batch, from image
    # Random crop
    # - specify boundaries
    # Random scale
    # - scale by 0.9 to 1.1 (default)
    # Random flip
    # - Horizontal
    # - vertical

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name='PSDs - No grass',
                    rawFolder='data/raw/PSD',
                    processFolder='data/processed/PSD_no_grass',
                    numShards=10,
                    data_url='https://vision.eng.au.dk/?download=/data/WeedData/Segmented.zip',
                    raw_filename='PSD_segmented.zip',
                    class_exclude_list=['Black-grass','Common wheat','Loose Silky-bent'],
                    max_img_dims=[400, 400], # [height, width]
                    ):
        super(Dataset, self).__init__(name = name,
                                        rawFolder = rawFolder, 
                                        processFolder = processFolder, 
                                        numShards = numShards,
                                        data_url=data_url,
                                        raw_filename=raw_filename)

        self.class_exclude_list = class_exclude_list
        self.max_img_dims = max_img_dims

    def _process_setup(self, rawFolder, processFolder):
        # Call parent class process_setup method
        super(Dataset, self)._process_setup(rawFolder, processFolder)

        # Delete folders matching class_exclude_list
        for class_exclude in self.class_exclude_list:
            shutil.rmtree(os.path.join(processFolder, 'Segmented', class_exclude))


        # TODO: Loop through all folders and images, and delete images with sizes greater than X
        data_root = os.path.join(processFolder,'Segmented')

        # Get subfolders
        class_directories = [path for path in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,path))]

        image_reader = self.ImageReader()
        for class_dir in class_directories:
            filenames = glob.glob(os.path.join(data_root, class_dir, '*.png'))
            for filename in filenames:
                img = image_reader.read(filename, tf_session=None)
                # Delete image, if it exceeds max dimensions
                if (img.shape[0] > self.max_img_dims[0]) or (img.shape[1] > self.max_img_dims[1]):
                    os.remove(filename)
                    

        return

    def _process_cleanup(self, processFolder):
        # Use for removing tmp files from process folder

        # TODO: Delete unzipped files
        return
