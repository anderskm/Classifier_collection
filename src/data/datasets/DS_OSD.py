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
                    name = 'OSD',
                    rawFolder = 'data/raw/OSD',
                    processFolder = 'data/processed/OSD',
                    numShards = 162):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _split_data_examples_to_shards(self, list_of_filenames, list_of_corresponding_class_names, list_of_unique_classes, num_shards, list_of_grouping_data):

        # list_of_lots = [os.path.split(os.path.split(filename)[0])[1] for filename in list_of_filenames]
        list_of_labs = [os.path.normpath(filename).split(os.sep)[3] for filename in list_of_filenames]
        labs = list(set(list_of_labs))
        labs.sort()

        list_of_labs_species = [os.path.normpath(filename).split(os.sep)[3:5] for filename in list_of_filenames]

        unique_labs_species = [list(x) for x in set(tuple(x) for x in list_of_labs_species)]

        unique_labs_species = sorted(unique_labs_species, key=itemgetter(0))

        shards_of_filenames = []
        shards_of_corresponding_class_names = []

        num_shards = []

        for lab in labs:
            lab_species = [s for l,s in unique_labs_species if l == lab]
            _num_shards = len(lab_species)*3*3
            num_shards.append(_num_shards)

            _list_of_filenames = [filename for filename, _lab in zip(list_of_filenames, list_of_labs) if _lab == lab]
            _list_of_corresponding_class_names = [class_names for class_names, _lab in zip(list_of_corresponding_class_names, list_of_labs) if _lab == lab]
            _list_of_grouping_data = [grouping_data for grouping_data, _lab in zip(list_of_grouping_data, list_of_labs) if _lab == lab]

            _shards_of_filenames, _shards_of_corresponding_class_names = super(Dataset, self)._split_data_examples_to_shards(_list_of_filenames, _list_of_corresponding_class_names, list_of_unique_classes, _num_shards, _list_of_grouping_data)

            shards_of_filenames += _shards_of_filenames
            shards_of_corresponding_class_names += _shards_of_corresponding_class_names

            fob = open(os.path.join(self.processFolder, 'lab_shard_idx.txt'),'w')
            shard_start = 1
            for lab, n_shards in zip(labs, num_shards):
                fob.write(lab + ', ' + ' '.join([str(i) for i in range(shard_start, shard_start+n_shards)]) + '\n')
                shard_start = shard_start+n_shards
            fob.close()

        return shards_of_filenames, shards_of_corresponding_class_names
