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

        def __init__(self):
            super(superDataset.Dataset.ImageReader, self).__init__()
            self.tf_raw_image = tf.placeholder(dtype=tf.uint8)
            self.tf_encoded_image = tf.image.encode_png(
                self.tf_raw_image,
                compression=-1,
                name='Image_encoder'
            )

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
            
            encoded_image = tf_session.run(self.tf_encoded_image,
                feed_dict={self.tf_raw_image: raw_image.astype('uint8')})
            return encoded_image, 'png'

        def decode(self, encoded_image):
            decoded_image = tf.image.decode_png(
                encoded_image,
                channels=1,
                dtype=tf.uint8,
                name='Image_decoder'
            )
            return decoded_image

        def to_RGB(self, raw_image, gain=1.0, normalize=True):
            red_channel_responses = np.asarray([0, 0, 0.0369139549956096, 0.0523482283247618, 0.105564094022188, 0.383669457929883, 0.730249295315904, 0.991933145444743, 0.930799922301415, 0.425092541012886, 0.2106692380065, 0.0975977243752676, 0.0068331963244273, 0, 0, 0, 0, 0, 0]).reshape((19,1))
            green_channel_responses = np.asarray([0, 0.0120732388216368, 0.0601790292541333, 0.0912116423383781, 0.189273593076368, 0.549788971492032, 0.918341352767597, 0.81765429467056, 0.507831521160346, 0.0710667549348757, 0.0289961859201456, 0.0124101353134283, 0.00338861996111386, 0, 0, 0, 0, 0, 0]).reshape((19,1))
            blue_channel_responses = np.asarray([0, 0.130978573502305, 0.893913954225284, 0.914091893437036, 0.590796359506189, 0.0920940594318619, 0.0244419235446471, 0.00491389861881971, 0.00491389861881971, 0.00491389861881971, 0.00432483677858752, 0.00432483677858752, 0, 0, 0, 0, 0, 0, 0]).reshape((19,1))
            raw_image = np.asarray(raw_image)

            red_channel = np.dot(raw_image, red_channel_responses)*gain
            green_channel = np.dot(raw_image, green_channel_responses)*gain
            blue_channel = np.dot(raw_image, blue_channel_responses)*gain

            if (normalize):
                red_channel /= red_channel_responses.sum()
                green_channel /= green_channel_responses.sum()
                blue_channel /= blue_channel_responses.sum()

            red_channel = np.minimum(255.0, red_channel)
            green_channel = np.minimum(255.0, green_channel)
            blue_channel = np.minimum(255.0, blue_channel)

            return np.concatenate((red_channel, green_channel, blue_channel), axis=-1).astype(dtype=np.uint8)

    def _filename_to_TFexample(self, filename):
        # Read the filename:
        imreader = self.ImageReader()
        # img_string = tf.io.read_file(filename)
        # raw_image = tf.image.decode_png(img_string)
        raw_image = imreader.read(filename)
        
        img_shape = tf.shape(raw_image)

        # Class
        # splits = tf.string_split([filename], "/")
        # splits = tf.string_split([splits.values[-1]],"\\")
        # classIdx = tf.string_to_number(splits.values[-2], out_type=tf.int64)
        classIdx = 0

        return raw_image, [classIdx], '-', img_shape[0], img_shape[1], img_shape[2], filename
