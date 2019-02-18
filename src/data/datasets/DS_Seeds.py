import src.data.Dataset as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Seeds Dataset',
                    rawFolder = 'data/raw/Seeds',
                    processFolder = 'data/processed/Seeds',
                    numShards = 10):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)      

    def _download(self):
        pass

    def _get_filenames_and_classes(self, raw_folder, process_folder):
        
        data_root = os.path.join(raw_folder)

        # Get filenames
        list_of_filenames = glob.glob(os.path.join(data_root,'*.tif'))

        # Extract class from filename
        list_of_corresponding_class_names = [None] * len(list_of_filenames)
        list_of_grouping_data = [None] * len(list_of_filenames)
        for i, filename in enumerate(list_of_filenames):
            list_of_corresponding_class_names[i] = self._filename_to_class(filename)
            list_of_grouping_data[i] = '1' #basename_parts[2]

        list_of_corresponding_class_names_T = list(map(list, zip(*list_of_corresponding_class_names))) # Transpose list of lists
        # print(list_of_corresponding_class_names_T)

        # list_of_unique_class_names = list(set(list_of_corresponding_class_names))
        list_of_unique_class_names = [list(set(class_names)) for class_names in list_of_corresponding_class_names_T]

        return list_of_filenames, list_of_corresponding_class_names, list_of_unique_class_names, list_of_grouping_data

    def _filename_to_class(self, filename):
        num2className = ['Not germinated', 'Germinated', 'Germinated', 'Not germinated', 'Germinated', 'Germinated']
        basename_without_extension = os.path.splitext(os.path.basename(filename))[0]
        basename_parts = basename_without_extension.split('_')
        class_this = num2className[int(basename_parts[8])-1] # Class for this image in image sequence
        class_end = num2className[int(basename_parts[9])-1] # Class at end of image sequence
        class_name = class_this
        return [class_name]
        
    class ImageReader(superDataset.Dataset.ImageReader):
        def read(self, filename, tf_session):
            encoded_img = PIL.Image.open(filename)
            img = np.empty(encoded_img.size, dtype=np.uint8)
            for i, page in enumerate(PIL.ImageSequence.Iterator(encoded_img)):
                if (i > 0): # Skip first channel/frame as it is an RGB version of the raw data
                    # Read image channel, convert to numpy array and append to rest of channels
                    pageDataAsNpArray = np.array(page.getdata(),np.uint8).reshape(encoded_img.size)
                    img = np.dstack((img,pageDataAsNpArray))
            # Remove the first "channel" as it is empty from the initialization
            img = np.delete(img,0,axis=2)
            self.numChannels = img.shape[2]
            
            return img.astype('uint8')

        def pack(self, raw_image, tf_session=None):
            height, width, channels = raw_image.shape
            packed_image = np.reshape(raw_image, (height,width*channels,1),order='F')
            return packed_image

        def unpack(self, packed_image, height, width, channels):

            unpacked_image = tf.transpose(packed_image,[1, 0, 2])
            unpacked_image = tf.reshape(unpacked_image,[channels, tf.math.multiply(height,width), 1])
            unpacked_image = tf.transpose(unpacked_image,[1,0,2])
            unpacked_image = tf.reshape(unpacked_image, [height, width, channels])
            unpacked_image = tf.transpose(unpacked_image,[1, 0, 2])

            return unpacked_image

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
                channels=1,
                dtype=tf.uint8,
                name='Image_decoder'
            )
            return decoded_image
