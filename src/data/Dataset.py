import datetime
import inspect
import itertools
import json
from functools import lru_cache
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time
import unittest


class Dataset(object):

    num_examples = None

    def _process_setup(self, rawFolder, processFolder):
        # Create processFolder directory
        if not os.path.exists(processFolder):
            os.makedirs(processFolder)
        
        # Use for e.g. unzipping images or copying them to process folder temporarily
        return

    def _process_cleanup(self, processFolder):
        # Use for removing tmp files from process folder
        return

    def _get_filenames_and_classes(self, rawFolder, processFolder):
        list_of_filenames = []
        list_of_corresponding_class_names = []
        list_of_unique_classes = []
        list_of_grouping_data = []
        raise NotImplementedError('Process method has not yet been implemented for this dataset.')
        return list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, list_of_grouping_data

    def _download(self):
        raise NotImplementedError('Download method has not yet been implemented for this dataset.')
    
    class ImageReader():
        def __init__(self):
            pass

        def read(self, filename, tf_session):
            # Reads an image from the harddrive
            # Note: Only used when creating tfrecords
            raw_image = np.empty([0,0])
            raise NotImplementedError('Read method of ImageReader has not yet been implemented for this dataset.')
            return raw_image

        def encode(self, raw_image, tf_session):
            # Encode raw image
            # Used for encoding images before storing them in tfrecords
            # Note: Speed optimization not nessecary
            return raw_image, 'raw'
        
        def decode(self, encoded_image):
            # Decode encoded image back into raw image
            # Used during traning and test
            # Note: Optimize for speed. Use tf methods
            return encoded_image

        def pack(self, unpacked_image, tf_session):
            # Pack image before encoding it.
            # E.g. reshaping multichannel image into 1-channel image
            # Used when storing images in tfrecords
            # Note: Speed optimization not nessecary
            packed_image = unpacked_image
            return packed_image
        
        def unpack(self, packed_image, height, width, channels):
            # Pack unimage after decoding it
            # E.g. reshaping multichannel image stored as 1-channel image into multichannels
            # Used when reading images from tfrecords
            # Note: Optimize for speed. Use tf methods
            unpacked_image = packed_image
            return unpacked_image

    def __init__(self, name = None, rawFolder = None, processFolder = None, numShards = None):
        # validation_method: holdout (train, validation, test), kfold (split into k equal folds), random (shuffle dataset and split random)
        # TODO: Variable inputs? kwargs
        self.name = name
        self.rawFolder = rawFolder
        self.processFolder = processFolder
        self.numShards = numShards

        # Create raw and processed folders, if they do not already exist
        if not os.path.exists(self.rawFolder):
            os.makedirs(self.rawFolder)
        if not os.path.exists(self.processFolder):
            os.makedirs(self.processFolder)

        print(self)

    def _encode_to_TFexample(self, image_data, image_format, class_lbl, height, width, channels = 0, class_text = None, origin = None):
        """ Encodes image data and label and returns a tfrecord example
        Args:
        image_data:   Encoded image (eg. tf.image.encode_png)
        image_format: Format in which the image is encoded
        class_lbl:    Class label to which the image belong
        height:       Image height
        width:        Image width
        channels:     Image channels
        class_text:   Readable class label, if not avaliable 
                        defaults to str(class_lbl)
        origin:       Filename of the original data file 
                        (Defaults to b'00', if unavailable)
        
        Returns:
        A tfrecord example
        """

        if class_text == None:
            class_text = str(class_lbl).encode()
        if origin == None:
            origin = '00'.encode()

        features = tf.train.Features(
            feature = {
                'image/encoded':    self.Features.bytes(image_data),
                'image/format':     self.Features.bytes(image_format.encode()),
                'image/class/label':self.Features.int64(class_lbl),
                'image/class/text': self.Features.bytes(class_text.encode()),
                'image/height':     self.Features.int64(height),
                'image/width':      self.Features.int64(width),
                'image/channels':   self.Features.int64(channels),
                'image/origin/filename': self.Features.bytes(origin.encode()),
            })

        return tf.train.Example(features = features)

    def _decode_from_TFexample(self, example_proto):
        """ decodes a tfrecord example and returns an image and label
        Args:
        example_proto: A tfrecord example
        
        Returns:
        image:      A decoded image tensor with type float32 
                    and shape [height, width, num_channels]. 
                    The image is normalized to be in range: 
                    -1.0 to 1.0   
        class_lbl:  Class label to which the image belong
        class_text: Readable class label
        height:     Image height
        width:      Image width
        channels:   Image channels
        origin:     Filename of the original data file 
                    (Defaults to b'00', if unavailable)
        """

        features = {
            'image/encoded':    tf.FixedLenFeature([], tf.string),
            'image/format':     tf.FixedLenFeature([], tf.string),
            'image/class/label':tf.VarLenFeature(tf.int64), #'image/class/label':tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/height':     tf.FixedLenFeature([], tf.int64),
            'image/width':      tf.FixedLenFeature([], tf.int64),
            'image/channels':    tf.FixedLenFeature([], tf.int64),
            'image/origin/filename': tf.FixedLenFeature([], tf.string)
        }

        # parsed_example = tf.parse_example(example_proto, features)
        parsed_example = tf.parse_single_example(example_proto, features)

        encoded_image = parsed_example['image/encoded']
        image_format = parsed_example['image/format']
        class_lbl = tf.sparse.to_dense(parsed_example['image/class/label'], default_value=-1)
        class_text = parsed_example['image/class/text']
        height = parsed_example['image/height']
        width = parsed_example['image/width']
        channels = parsed_example['image/channels']
        origin = parsed_example['image/origin/filename']

        image_reader = self.ImageReader()
        tf_decoded_image = image_reader.decode(encoded_image)
        tf_unpacked_image = image_reader.unpack(tf_decoded_image, height, width, channels)

        return tf_unpacked_image, class_lbl, class_text, height, width, channels, origin

    def _convert_to_tfrecord(self, listOfFilenames, listOfCorrespondingClasses, class_dicts, tfrecord_writer, tf_session):
        """Loads data from the binary MNIST files and writes files to a TFRecord.

        Args:
            data_filename: The filename of the MNIST images.
            labels_filename: The filename of the MNIST labels.
            num_images: The number of images in the dataset.
            tfrecord_writer: The TFRecord writer to use for writing.
        """
        
        num_images = len(listOfFilenames)

        image_reader = self.ImageReader()

        for i in range(num_images):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, num_images))
            sys.stdout.flush()

            # Read the filename:
            raw_image = image_reader.read(listOfFilenames[i], tf_session)
            height, width, channels = raw_image.shape
            # Pack and encode image
            packed_image = image_reader.pack(raw_image, tf_session)
            encoded_image, encode_format = image_reader.encode(packed_image, tf_session)
            # Get image labels
            class_names = listOfCorrespondingClasses[i]
            labels = [class_dict[class_name] for class_dict,class_name in zip(class_dicts, class_names)]

            class_text = ''
            for class_name in class_names:
                class_text += class_name + ';' 
            class_text = class_text[0:-1]

            # Create tensorflow example
            tf_example = self._encode_to_TFexample(encoded_image, encode_format, labels, height, width, channels = channels, class_text = class_text, origin = listOfFilenames[i])

            # Write example to TFrecord
            tfrecord_writer.write(tf_example.SerializeToString())
        print('\n')

    def _split_data_examples_to_shards(self, list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, num_shards, list_of_grouping_data):
        shards_of_filenames = []
        shards_of_corresponding_class_names = []
        for _ in range(num_shards):
            shards_of_filenames.append([])
            shards_of_corresponding_class_names.append([])

        # Merge filenames and classes and switch dimensions
        list_of_filenames_and_classes = [list_of_filenames, list_of_corresponding_class_names]
        list_of_filenames_and_classes = list(map(list, zip(*list_of_filenames_and_classes))) # "Transpose" list of lists (from [[filenames],[classes]] to [[filename1, class1], ...])

        # Shuffle list before sharding --> Assume, that classes are then split (approximately) evenly across shards
        rand_state = random.getstate() # Store current state of random generator
        random.seed(1337) # Set fixed seed before splitting dataset to ensure reproduceability
        random.shuffle(list_of_filenames_and_classes) # Shuffle is done in-place
        random.setstate(rand_state) # Resote previous state of the random generator

        # Treat each class individually to spred them (approximately) equally across shards
        # for unique_class in list_of_unique_classes:
            # Get all data examples, which belong to this class
            # list_of_filenames_and_classes_for_unique_class = [x for x in list_of_filenames_and_classes if x[1] == unique_class]
            # Split into shards
            # shards = [list_of_filenames_and_classes_for_unique_class[i::num_shards] for i in iter(range(num_shards))]
        shards = [list_of_filenames_and_classes[i::num_shards] for i in iter(range(num_shards))]
        # Process each shard
        for shard_n in range(num_shards):
            shard = list(map(list, zip(*shards[shard_n]))) # Transpose list of lists (from [[filename1, class1], ...] to [[filenames],[classes]])
            shards_of_filenames[shard_n].extend(shard[0]) # shard[0] = filenames
            shards_of_corresponding_class_names[shard_n].extend(shard[1]) # shard[1] = classes

        return shards_of_filenames, shards_of_corresponding_class_names

    def download(self):
        self._download()

    def process(self):
        # Setup dataset process
        self._process_setup( self.rawFolder, self.processFolder)

        list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, list_of_grouping_data = self._get_filenames_and_classes(self.rawFolder, self.processFolder)

        class_dicts = [dict(zip(unique_classes, range(len(unique_classes)))) for unique_classes in list_of_unique_classes]
        
        # self._save_dict(list_of_unique_classes, self.processFolder, 'class_dict.json')
        self._save_dict(class_dicts, self.processFolder, 'class_dict.json')

        # Split data examples into shards
        shards_of_filenames, shards_of_corresponding_class_names = self._split_data_examples_to_shards(list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, self.numShards, list_of_grouping_data)
        
        with tf.Session('') as tf_session:
            for shard_n in range(self.numShards):
                self._show_message('Processing shard %d/%d' % (shard_n+1,self.numShards))
                tf_filename = self._get_output_filename(self.processFolder, shard_n, self.numShards)

                with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                    # TODO: Calculate per-pixel mean across dataset and save
                    # Only for training set???
                    self._convert_to_tfrecord(shards_of_filenames[shard_n], shards_of_corresponding_class_names[shard_n], class_dicts, tfrecord_writer, tf_session)

        self._process_cleanup(self.processFolder)

    def get_dataset_list(self, tf_session=None, data_source = 'tfrecords', data_folder = None, shuffle_before_split=True, shuffle_seed=1337, group_before_split=False, validation_method='none', holdout_split=[0.8, 0.1, 0.1], cross_folds=10, cross_val_fold=None, cross_test_fold=0, shard_val=None, shard_test=[0], stratify_training_set = True):
        # Return af list of the datasets split according to the chosen validation method. E.g. [train, val, test] for holdout

        close_session = False
        if (tf_session is None):
            tf_session = tf.Session('')
            close_session = True

        # TODO: Expose following variables to user
        # shuffle_before_split = True
        # shuffle_seed = 1337 # Sets the operation seed. See documentation for tf.random.set_random_seed (https://www.tensorflow.org/api_docs/python/tf/random/set_random_seed)

        # group_before_split = True

        # validation_method = 'holdout'

        # holdout_split = [0.8, 0.1, 0.1] # It is normalized before the split

        # validation_method = 'shard'
        # shards_train = []
        # shards_val = []
        # shards_test = []

        cross_folds = 10 # Number of folds in cross validation
        cross_val_folds = [] # Index of the fold used for validation/test
        cross_test_folds = [0] # Index of the fold used for validation/test

        random_split = [0.9, 0.1]

        # stratify_training_set = True

        # TODO:
        #  Grouping using group_by_reducer or group_by_window:
        #   https://www.tensorflow.org/api_docs/python/tf/data/experimental/group_by_reducer
        #   https://www.tensorflow.org/api_docs/python/tf/data/experimental/group_by_window
        #   Treat each group as a single example, split into train, test and val, then ungroup using data.Dataset.flat_map()?
        #
        # Split tensorflow dataset into test and train
        # https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
        #
        # Cross-validation
        # - K-fold cross validation
        # -- stratified --> keep same class distribution within each fold
        # -- merge datasets: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#zip
        # - holdout --> split into training, validation and test set once
        # - monte carlo --> random split for each fold
        # - none --> return full dataset

        # When validation method = shard(s) overwrite settings
        if (validation_method == 'shards'):
            group_before_split = False
            shuffle_before_split = False

        with tf.name_scope('Dataset_manager'):

            if (data_source == 'tfrecords'):
                # Get all tfrecords
                dataset_filenames = [self._get_output_filename(self.processFolder, i, self.numShards) for i in range(self.numShards)]

                # Create full dataset from tfrecords
                if (validation_method == 'shards'):
                    pass
                else:
                    tf_dataset = tf.data.TFRecordDataset(dataset_filenames)
            elif (data_source == 'folder'):
                tf_dataset = tf.data.Dataset.list_files(data_folder, shuffle=False)
                tf_dataset = tf_dataset.map(lambda origin: self._filename_to_TFexample( origin))


            # Group
            if (group_before_split):
                tf_dataset_name = tf_dataset.map(lambda example,: self._get_name_only(example))
                tf_dataset_id = tf_dataset_name.map(lambda example, origin: self._name_to_ID(example, origin))

                tf_dataset = tf_dataset_id.apply( tf.data.experimental.group_by_window(
                                                                                    key_func= lambda example, id: id,
                                                                                    reduce_func=lambda id, ds_group: ds_group.map(lambda example, _: example).batch(10000),
                                                                                    window_size=10000
                                                                                        )
                                                    )
                ## Uncomment for testing
                # tf_dataset_iterator = tf_dataset.make_one_shot_iterator()
                # tf_input_getBatch = tf_dataset_iterator.get_next()
                # for i in range(10):
                #     this_batch = tf_session.run(tf_input_getBatch)
            
            if (shuffle_before_split):
                # Sets the operation seed. See documentation for tf.random.set_random_seed (https://www.tensorflow.org/api_docs/python/tf/random/set_random_seed)
                tf_dataset = tf_dataset.shuffle(buffer_size=10000, seed=shuffle_seed, reshuffle_each_iteration=False)

            print('Dataset validation method: ' + validation_method)

            if validation_method == 'none':
                tf_dataset_list = [tf_dataset, None, tf_dataset]
            elif validation_method == 'holdout':
                holdout_split = holdout_split/np.sum(holdout_split) # Normalize

                dataset_size = self._get_num_examples(tf_dataset, tf_session)
                train_size = int(holdout_split[0] * dataset_size)
                val_size = int(holdout_split[1] * dataset_size)
                test_size = int(dataset_size - train_size - val_size)

                with tf.name_scope('Train'):
                    tf_dataset_train = tf_dataset.take(train_size)
                tf_dataset_remainder = tf_dataset.skip(train_size)
                with tf.name_scope('Validation'):
                    tf_dataset_val = tf_dataset_remainder.take(val_size)
                with tf.name_scope('Test'):
                    tf_dataset_test = tf_dataset_remainder.skip(val_size)
                # tf_dataset_test = tf_dataset_test.take(test_size)

                tf_dataset_list = [tf_dataset_train, tf_dataset_val, tf_dataset_test]
            elif (validation_method == 'shards'):
                dataset_filenames_train = dataset_filenames.copy() # Make a an actual copy of the list and not just the reference to the list
                # Set validation set
                if (shard_val == None) or (len(shard_val) < 0):
                    tf_dataset_val = None
                else:
                    dataset_filenames_val = dataset_filenames[shard_val]
                    tf_dataset_val = tf.data.TFRecordDataset(dataset_filenames_val)
                    print('Validation shard:')
                    print(dataset_filenames_val)
                    dataset_filenames_train.remove(dataset_filenames_val)
                # Set test set
                if (shard_test == None) or (len(shard_test) < 0):
                    tf_dataset_test = None
                else:
                    dataset_filenames_test = []
                    for this_shard_test in shard_test:
                        dataset_filenames_test.append(dataset_filenames[this_shard_test])
                        dataset_filenames_train.remove(dataset_filenames[this_shard_test])
                    print('Test shard(s):')
                    print(dataset_filenames_test)
                    tf_dataset_test = tf.data.TFRecordDataset(dataset_filenames_test)
                # Set training set
                print('Training shard:')
                print(dataset_filenames_train)
                tf_dataset_train = tf.data.TFRecordDataset(dataset_filenames_train)
                
                tf_dataset_list = [tf_dataset_train, tf_dataset_val, tf_dataset_test]
            else:
                raise ValueError('Unknown validation method: ' + validation_method)
            
            # Ungroup
            if (group_before_split):
                tf_dataset_list = [tf_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) for tf_dataset in tf_dataset_list]

            print('Dataset splits:')
            dataset_sizes = [self._get_num_examples(_tf_dataset, tf_session) for _tf_dataset in tf_dataset_list]
            [print('   ' + '{:>4.3f}'.format(dataset_size/sum(dataset_sizes)) + '{:>7d}'.format(dataset_size)) for dataset_size in dataset_sizes]
            # print('   Training  : ' + str(holdout_split[0]) + '    ' + str(train_size))
            # print('   Validation: ' + str(holdout_split[1]) + '    ' + str(val_size))
            # print('   Test      : ' + str(holdout_split[2]) + '    ' + str(test_size))

            if (stratify_training_set):
                print(' ')
                print('Stratify training set')

                # Grab training dataset from list of datasets
                tf_dataset_train = tf_dataset_list[0]

                # Determine number of outputs and classes per output
                class_dicts = self.get_class_dicts()
                num_classes = [len(class_dict) for class_dict in class_dicts]
                print('Number of classes per output : ' + ', '.join([str(x) for x in num_classes]))
                # Determine number of potential output-class combinations
                num_class_combinations = np.prod(num_classes)
                print('Number of output combinations: ' + str(num_class_combinations))

                # Create list of potential class combinations
                class_lists = [[i for i in range(N_classes)] for N_classes in num_classes]
                class_combinations = [list(x) for x in itertools.product(*class_lists)]
                
                # Batch examples according to class combination
                tf_dataset_key = tf_dataset_train.map(lambda example: self._class_combination_to_key(example, class_combinations))
                tf_dataset_classes = tf_dataset_key.apply( tf.data.experimental.group_by_window(
                                                                                    key_func= lambda example, lbl, key: key,
                                                                                    reduce_func=lambda id, ds_group: ds_group.map(lambda example, _, __: example).batch(10000),
                                                                                    window_size=10000
                                                                                        )
                                                    )
                # Get number of class combinations generated (all theoretical combinations might not be generated)
                num_class_combinations_in_dataset = self._get_num_examples(tf_dataset_classes, tf_session)

                # Split class combinations into seperat datasets
                tf_dataset_classes_list = []
                tf_dataset_remainder = tf_dataset_classes
                for i in range(num_class_combinations_in_dataset):
                    # Grab one class combination and "undo" batching by using flat_map
                    tf_dataset_classes_list.append(tf_dataset_remainder.take(1).flat_map(lambda serialized_example: tf.data.Dataset.from_tensor_slices(serialized_example)))
                    tf_dataset_remainder = tf_dataset_remainder.skip(1)
                
                # Get size of each class-combination dataset
                dataset_class_sizes = [self._get_num_examples(tf_dataset_class, tf_session) for tf_dataset_class in tf_dataset_classes_list]
                print(dataset_class_sizes)
                # Calculate scaling factor, such that each class-combination has approximately equal representation
                scale_factors = np.round(np.max(dataset_class_sizes)/np.asarray(dataset_class_sizes)).astype(int)
                print(scale_factors)
                # Scale each dataset by its scale factor
                for i,tf_dataset_class in enumerate(tf_dataset_classes_list):
                    # Two methods for repeating the same dataset multiple times.
                    # Both methods seem to take about the same amount of time during training.

                    ## Method 1
                    # tf_dataset_class_tmp = tf_dataset_class
                    # for k in range(scale_factors[i]-1):
                    #     tf_dataset_class_tmp = tf_dataset_class_tmp.concatenate(tf_dataset_class)
                    # tf_dataset_classes_list[i] = tf_dataset_class_tmp

                    ## Method 2:
                    tf_dataset_classes_list[i] = tf_dataset_class.repeat(count=scale_factors[i])

                # Print number of examples in each class-combination dataset
                print([self._get_num_examples(tf_dataset_class, tf_session) for tf_dataset_class in tf_dataset_classes_list])
                
                # Merge class-combination datasets into a single dataset
                tf_dataset_train_stratified = tf_dataset_classes_list[0]
                for tf_dataset_class in tf_dataset_classes_list[1:]:
                    tf_dataset_train_stratified = tf_dataset_train_stratified.concatenate(tf_dataset_class)

                # Print total number of exambles in stratified dataset
                print(self._get_num_examples(tf_dataset_train_stratified, tf_session))

                tf_dataset_list[0] = tf_dataset_train_stratified
                
                # Update dataset sizes
                dataset_sizes = [self._get_num_examples(_tf_dataset, tf_session) for _tf_dataset in tf_dataset_list]

        if (close_session):
            print('Closing tf session...')
            tf_session.close()

        return tf_dataset_list, dataset_sizes

    def save_dataset_filenames(self, filename, tf_dataset, tf_session=None):
        close_session = False
        if (tf_session is None):
            print('TF session not specified. Creating new tf session.')
            tf_session = tf.Session('')
            close_session = True
        
        fobj = open(filename, 'w')

        try:
            tf_dataset_name = tf_dataset.map(lambda example,: self._get_name_only2(example))
            tf_dataset_name_iterator = tf_dataset_name.make_one_shot_iterator()
            tf_input_getBatch = tf_dataset_name_iterator.get_next()
        
            while True:
                this_batch = tf_session.run(tf_input_getBatch)
                fobj.write(this_batch.decode('utf-8') + '\n')
        except tf.errors.OutOfRangeError:
            pass
        
        fobj.close()
        if (close_session):
            print('Closing tf session...')
            tf_session.close()

    
    def _filename_to_TFexample(self, filename):
        raise NotImplementedError('Method for converting an image file to tf example has not been implemented for this dataset.')
        return image, class_idx, class_name, height, width, channels, filename
        
    def _class_combination_to_key(self, example_proto, class_combinations):
        
        features = {'image/class/label': tf.VarLenFeature(tf.int64)}
        parsed_example = tf.parse_single_example(example_proto, features)
        class_lbl = tf.sparse.to_dense(parsed_example['image/class/label'], default_value=-1)

        key = tf.argmax(tf.cast(tf.reduce_all(tf.equal(tf.cast(class_combinations,tf.int64), class_lbl),axis=1), tf.int64), axis=0) #tf.argmax(tf.cast(tf.equal(tf.cast(class_combinations,tf.int64), class_lbl), tf.int64), axis=1)        

        return example_proto, class_lbl, key

    def _get_name_only2(self, example_proto):
         # Get filename only
        features = {
            'image/origin/filename': tf.FixedLenFeature([], tf.string)
        }

        parsed_example = tf.parse_single_example(example_proto, features)

        origin = parsed_example['image/origin/filename']

        return origin

    def _get_name_only(self, example_proto):
         # Get filename only
        features = {
            'image/origin/filename': tf.FixedLenFeature([], tf.string)
        }

        parsed_example = tf.parse_single_example(example_proto, features)

        origin = parsed_example['image/origin/filename']

        return example_proto, origin

    def _name_to_ID(self, example_proto, origin):
        # TODO: Move to function to DS_Seeds. Function is too case specific
        # TODO: Add generic name_to_ID using tf.string_to_hash_bucket_fast

        # Remove path from filename
        splits = tf.string_split([origin], "/")
        splits = tf.string_split([splits.values[-1]],"\\")
        basename = splits.values[-1]
        # Remove file extension from basename
        splits = tf.string_split([basename],'.')
        basename = splits.values[0:-1]

        # Get metadata from basename
        splits = tf.string_split(basename,'_')

        ageing_split = tf.string_split([splits.values[1]], '')
        ageing = tf.string_to_number(ageing_split.values[0], out_type=tf.int64)
        primed = tf.cond(tf.equal(splits.values[2],'NP'), lambda: tf.to_int64(0), lambda: tf.to_int64(1))
        repetition_split = tf.string_split([splits.values[3]], '')
        repetition = tf.string_to_number(repetition_split.values[1], out_type=tf.int64)
        seed_id = tf.string_to_number(splits.values[5], out_type=tf.int64)

        return example_proto, repetition*10000+primed*1000+ageing*100+seed_id

    @lru_cache(maxsize=128)
    def _get_num_examples(self, tf_dataset, tf_session):
        if (tf_dataset is None):
            example_counter = 0
        else:
            close_session = False
            if (tf_session is None):
                tf_session = tf.Session('')
                close_session = True
            
            tf_dataset_count = tf_dataset.batch(1)
            tf_dataset_count_iterator = tf_dataset_count.make_initializable_iterator()
            tf_input_example = tf_dataset_count_iterator.get_next()
            tf_session.run(tf_dataset_count_iterator.initializer) # Initialize iterator
            example_counter = 0
            while True:
                try:
                    tmp = tf_session.run(tf_input_example)
                    example_counter += 1
                except tf.errors.OutOfRangeError:
                    # Do some evaluation after each Epoch
                    break
            if (close_session):
                print('Closing tf session...')
                tf_session.close()

        return example_counter

    def get_class_dicts(self):
        class_dict = self._load_dict(self.processFolder, 'class_dict.json')
        return class_dict

    #############
    # Unit test #
    #############

    def run_unit_tests_quick(self):
        DatasetUnitTest = self.UnitTestQuick
        DatasetUnitTest.dataset = self # Pass the current Dataset class to the unit test
        testLoader = unittest.TestLoader()
        testLoader.sortTestMethodsUsing = None
        suite = testLoader.loadTestsFromTestCase(DatasetUnitTest)
        unittest.TextTestRunner(verbosity=2).run(suite)
    
    def run_unit_tests_full(self):

        self.run_unit_tests_quick()

        # Run full unit tests
        DatasetUnitTest = self.UnitTestFull
        DatasetUnitTest.dataset = self # Pass the current Dataset class to the unit test
        testLoader = unittest.TestLoader()
        testLoader.sortTestMethodsUsing = None
        suite = testLoader.loadTestsFromTestCase(DatasetUnitTest)
        unittest.TextTestRunner(verbosity=2).run(suite)

    class UnitTestQuick(unittest.TestCase):

        dataset = None

        @classmethod
        def setUpClass(cls):
            # config = tf.ConfigProto(
            #             device_count = {'GPU': 0} # Force test to be run on CPU
            #         )
            # cls.tf_session = tf.Session(config=config)
            cls.tf_session = tf.Session()

            cls.dataset._process_setup(cls.dataset.rawFolder, cls.dataset.processFolder)

        @classmethod
        def tearDownClass(cls):
            cls.dataset._process_cleanup(cls.dataset.processFolder)
            cls.tf_session.close()

        def test_get_filenames_and_classes(self):
            DS = self.dataset
            list_of_filenames, list_of_corresponding_classNames, list_of_unique_classes, list_of_grouping_data = DS._get_filenames_and_classes(DS.rawFolder, DS.processFolder)

            # Test
            self.assertEqual(len(list_of_filenames), len(list_of_corresponding_classNames))
            self.assertEqual(len(list_of_filenames), len(list_of_grouping_data))
            # self.assertListEqual(list(set(list_of_corresponding_classNames)),list_of_unique_classes)


        def test_pack_unpack_one_example(self, filename=None, tf_session=None):
            if (filename is None):
                DS = self.dataset
                list_of_filenames, list_of_corresponding_classNames, list_of_unique_classes, list_of_grouping_data = DS._get_filenames_and_classes(DS.rawFolder, DS.processFolder)
                filename = list_of_filenames[0]

            if (tf_session is None):
                tf_session = self.tf_session

            # Setup dataset image reader
            image_reader = self.dataset.ImageReader()
            # Read image
            raw_image = image_reader.read(filename, tf_session)
            height, width, channels = raw_image.shape
            # Pack image
            packed_image = image_reader.pack(raw_image, tf_session)
            # Unpack image
            tf_packed_image = tf.placeholder(dtype=tf.uint8)
            tf_image_height = tf.placeholder(dtype=tf.int64)
            tf_image_width = tf.placeholder(dtype=tf.int64)
            tf_image_channels = tf.placeholder(dtype=tf.int64)
            tf_unpacked_image = image_reader.unpack(tf_packed_image, tf_image_height, tf_image_width, tf_image_channels)
            unpacked_image = tf_session.run(tf_unpacked_image,
                                            feed_dict={tf_packed_image: packed_image.astype('uint8'), 
                                                        tf_image_height: height,
                                                        tf_image_width: width,
                                                        tf_image_channels: channels})
            # Test intermediate packed shape. Third dim must be 1, 2, 3 or 4 (for png, for jpeg it must be 1 or 3)
            self.assertIn(packed_image.shape[2], [1,2,3,4])
            # Test for equality
            np.testing.assert_array_equal(raw_image, unpacked_image)

        def test_encode_decode_one_example(self, filename=None, tf_session=None):
            if (filename is None):
                DS = self.dataset
                list_of_filenames, list_of_corresponding_classNames, list_of_unique_classes, list_of_grouping_data = DS._get_filenames_and_classes(DS.rawFolder, DS.processFolder)
                filename = list_of_filenames[0]

            if (tf_session is None):
                tf_session = self.tf_session
            
            # Setup dataset image reader
            image_reader = self.dataset.ImageReader()
            # Read image
            raw_image = image_reader.read(filename, tf_session)
            height, width, channels = raw_image.shape
            # Pack image
            packed_image = image_reader.pack(raw_image, tf_session)
            # Encode image
            encoded_image, encode_format = image_reader.encode(packed_image, tf_session)

            # Setup decode and unpack
            tf_encoded_image = tf.placeholder(dtype=tf.string)
            # tf_packed_image = tf.placeholder(dtype=tf.uint8)
            tf_image_height = tf.placeholder(dtype=tf.int64)
            tf_image_width = tf.placeholder(dtype=tf.int64)
            tf_image_channels = tf.placeholder(dtype=tf.int64)

            tf_decoded_image = image_reader.decode(tf_encoded_image)
            tf_unpacked_image = image_reader.unpack(tf_decoded_image, tf_image_height, tf_image_width, tf_image_channels)

            decoded_image, unpacked_image = tf_session.run([tf_decoded_image, tf_unpacked_image],
                                                            feed_dict={tf_encoded_image: encoded_image,
                                                                        tf_image_height: height,
                                                                        tf_image_width: width,
                                                                        tf_image_channels: channels})

            # Test for equality
            np.testing.assert_array_equal(packed_image, decoded_image)
            np.testing.assert_array_equal(raw_image, unpacked_image)

    class UnitTestFull(unittest.TestCase):

        dataset = None

        @classmethod
        def setUpClass(cls):
            # config = tf.ConfigProto(
            #             device_count = {'GPU': 0} # Force test to be run on CPU
            #         )
            # cls.tf_session = tf.Session(config=config)
            cls.tf_session = tf.Session('')
        
            cls.dataset._process_setup(cls.dataset.rawFolder, cls.dataset.processFolder)

        @classmethod
        def tearDownClass(cls):
            cls.dataset._process_cleanup(cls.dataset.processFolder)
            cls.tf_session.close()

        def test_encode_decode_all_examples(self, tf_session=None):
            DS = self.dataset
            list_of_filenames, list_of_corresponding_classNames, list_of_unique_classes, list_of_grouping_data = DS._get_filenames_and_classes(DS.rawFolder, DS.processFolder)
            
            if (tf_session is None):
                tf_session = self.tf_session

            # Setup dataset image reader
            image_reader = self.dataset.ImageReader()
             # Setup decode and unpack
            tf_encoded_image = tf.placeholder(dtype=tf.string)
            tf_image_height = tf.placeholder(dtype=tf.int64)
            tf_image_width = tf.placeholder(dtype=tf.int64)
            tf_image_channels = tf.placeholder(dtype=tf.int64)
            tf_decoded_image = image_reader.decode(tf_encoded_image)
            tf_unpacked_image = image_reader.unpack(tf_decoded_image, tf_image_height, tf_image_width, tf_image_channels)

            # Loop through all images and test them
            sys.stdout.write('\nTesting image')
            for i, filename in enumerate(list_of_filenames):
                with self.subTest(msg=str(i) + '/' + str(len(list_of_filenames)) + ' - ' + filename):
                    sys.stdout.write('\r>> Testing image %d/%d' % (i + 1, len(list_of_filenames)))
                    sys.stdout.flush()

                    # Read image
                    raw_image = image_reader.read(filename, tf_session)
                    height, width, channels = raw_image.shape
                    # Pack image
                    packed_image = image_reader.pack(raw_image, tf_session)
                    # Encode image
                    encoded_image, encode_format = image_reader.encode(packed_image, tf_session)

                    # Decode and unpack image
                    decoded_image, unpacked_image = tf_session.run([tf_decoded_image, tf_unpacked_image],
                                                                    feed_dict={tf_encoded_image: encoded_image,
                                                                                tf_image_height: height,
                                                                                tf_image_width: width,
                                                                                tf_image_channels: channels})
                    
                    # Test for equality per image
                    np.testing.assert_array_equal(packed_image, decoded_image)
                    np.testing.assert_array_equal(raw_image, unpacked_image)

                sys.stdout.flush()

            

    ######
    # Features
    ######

    class Features(object):
        @staticmethod
        def int64(values):
            """Returns a TF-Feature of int64s.
            Args:
            values: A scalar or list of values.
            
            Returns:
            A TF-Feature.
            """
            if not isinstance(values, (tuple, list)):
                values = [values]
            return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        @staticmethod
        def bytes(values):
            """Returns a TF-Feature of bytes.
            
            Args:
            values: A string.

            Returns:
            A TF-Feature.
            """
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

        @staticmethod
        def float(values):
            """Returns a TF-Feature of floats.

            Args:
            values: A scalar of list of values.

            Returns:
            A TF-Feature.
            """
            if not isinstance(values, (tuple, list)):
                values = [values]
            return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    ######
    # Utility functions
    ######

    def _show_message(self, msg_str, lvl=0):

        if lvl == 0:
            print(datetime.datetime.now(), '-', msg_str)
        elif lvl == 1:
            print('______________________________________________________________')
            print(datetime.datetime.now(), '-', msg_str)
            print('--------------------------------------------------------------')
        else:
            pass

    def _save_dict(self, list_of_dicts, path, filename):
        fullpath = os.path.join(path, filename)
        with open(fullpath, 'w+') as fp:
            # for this_dict in list_of_dicts:
            json.dump(list_of_dicts, fp)

    def _load_dict(self, path, filename):
        fullpath = os.path.join(path, filename)
        with open(fullpath, 'r') as fp:
            loaded_dict = json.load(fp)
        return loaded_dict

    def _get_output_filename(self, dataset_dir, shard_id, numShards):
        """Creates the output filename.

        Args:
        dataset_dir: The directory where the temporary files are stored.
        split_name: The name of the train/test split.

        Returns:
        An absolute file path.
        """
        return '%s/data_shard_%03d-of-%03d.tfrecord' % (dataset_dir, shard_id+1, numShards)

    def __str__(self):
        return 'DATASET\n Name            : ' + self.name + '\n Raw folder      : ' + self.rawFolder + '\n Processed folder: ' + self.processFolder + '\n Num. shards     : ' + str(self.numShards)
