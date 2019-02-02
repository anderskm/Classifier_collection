import src.data.Dataset as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

import sys
from six.moves import urllib
import zipfile

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
                    name = 'Plant Seedling Dataset - Segmented',
                    rawFolder = 'data/raw/PSD',
                    processFolder = 'data/processed/PSD',
                    numShards = 10,
                    data_url='https://vision.eng.au.dk/?download=/data/WeedData/Segmented.zip',
                    raw_filename = 'PSD_segmented.zip'):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

        self._data_url = data_url
        self._raw_filename = raw_filename

    def download(self):
        """Downloads PSD locally
        """
        data_url = self._data_url
        filepath = os.path.join(self.rawFolder, self._raw_filename)
        
        if not os.path.exists(filepath):
            print('Downloading dataset...')
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %.1f%%' % (
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)

            print()
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print('Successfully downloaded', size, 'bytes.')
        else:
            print('Dataset has already been downloaded.')
            print('(Expected file already exist: ' + filepath + ')')

    def _process_setup(self, rawFolder, processFolder):
        # Locate zip-file in raw folder
        data_filename = glob.glob(os.path.join(rawFolder,'*zip'))[0]
        # Unzip images and store them in the processed data folder
        archive = zipfile.ZipFile(data_filename)
        archive.extractall(processFolder)
        
        return

    def _process_cleanup(self, processFolder):
        # Use for removing tmp files from process folder

        # TODO: Delete unzipped files
        return

    def _get_filenames_and_classes(self, raw_folder, process_folder):
        
        data_root = os.path.join(process_folder,'Segmented')

        # Get subfolders
        class_directories = [path for path in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,path))]

        list_of_filenames = []
        list_of_corresponding_class_names = []

        for class_dir in class_directories:
            filenames = glob.glob(os.path.join(data_root, class_dir, '*.png'))
            list_of_filenames += filenames
            list_of_corresponding_class_names += [[class_dir] for f in filenames]

        list_of_grouping_data = [None for i in range(len(list_of_filenames))]


        # Get filenames
        # list_of_filenames = glob.glob(os.path.join(data_root,'*.tif'))

        # # Extract class from filename
        # list_of_corresponding_class_names = [None] * len(list_of_filenames)
        # list_of_grouping_data = [None] * len(list_of_filenames)
        # for i, filename in enumerate(list_of_filenames):
        #     list_of_corresponding_class_names[i] = self._filename_to_class(filename)
        #     list_of_grouping_data[i] = '1' #basename_parts[2]

        list_of_corresponding_class_names_T = list(map(list, zip(*list_of_corresponding_class_names))) # Transpose list of lists
        # # print(list_of_corresponding_class_names_T)

        # # list_of_unique_class_names = list(set(list_of_corresponding_class_names))
        list_of_unique_class_names = [list(set(class_names)) for class_names in list_of_corresponding_class_names_T]

        return list_of_filenames, list_of_corresponding_class_names, list_of_unique_class_names, list_of_grouping_data
        
    class ImageReader(superDataset.Dataset.ImageReader):
        def read(self, filename, tf_session):
            encoded_img = PIL.Image.open(filename)
            img = np.asarray(encoded_img, dtype=np.uint8)
            return img

        def encode(self, raw_image, tf_session=None):
            tf_raw_image = tf.placeholder(dtype=tf.uint8)
            tf_encoded_image = tf.image.encode_png(
                tf_raw_image,
                compression=-1,
                name='Image_encoder'
            )
            encoded_image = tf_session.run(tf_encoded_image,
                feed_dict={tf_raw_image: raw_image.astype('uint8')})
            return encoded_image, 'png'

        def decode(self, encoded_image):
            decoded_image = tf.image.decode_png(
                encoded_image,
                channels=3,
                dtype=tf.uint8,
                name='Image_decoder'
            )
            return decoded_image
