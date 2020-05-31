import src.data.datasets.DS_OSD as superDataset
import os
import random
# import numpy as np
# import PIL
# import tensorflow as tf
# import os
# import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, Randers',
                    rawFolder = 'data/raw/OSD_Ra',
                    processFolder = 'data/processed/OSD_Ra',
                    numShards = 45):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _split_data_examples_to_shards(self, list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, num_shards, list_of_grouping_data):

        list_of_lots = [os.path.split(os.path.split(filename)[0])[1] for filename in list_of_filenames]

        unique_lots = list(set(list_of_lots))
        unique_lots.sort()

        _num_shards = int(num_shards / len(unique_lots))

        shards_of_filenames = []
        shards_of_corresponding_class_names = []

        for lot in unique_lots:
            _list_of_filenames = [filename for filename, _lot in zip(list_of_filenames, list_of_lots) if _lot == lot]
            _list_of_corresponding_class_names = [class_names for class_names, _lot in zip(list_of_corresponding_class_names, list_of_lots) if _lot == lot]
            _list_of_grouping_data = [grouping_data for grouping_data, _lot in zip(list_of_grouping_data, list_of_lots) if _lot == lot]

            _shards_of_filenames, _shards_of_corresponding_class_names = super(Dataset, self)._split_data_examples_to_shards(_list_of_filenames, _list_of_corresponding_class_names, list_of_unique_classes, _num_shards, _list_of_grouping_data)

            shards_of_filenames += _shards_of_filenames
            shards_of_corresponding_class_names += _shards_of_corresponding_class_names

        return shards_of_filenames, shards_of_corresponding_class_names
