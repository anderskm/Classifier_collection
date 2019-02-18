import src.data.datasets.DS_Okra as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'Okra seed dataset, D0',
                    rawFolder = 'data/raw/Okra',
                    processFolder = 'data/processed/Okra_D0',
                    numShards = 1):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _filename_to_class(self, filename):
        # Extrac class names from filename
        classes = self._filename_to_classes(filename)
        # Grab classes of the last image for this image in the image sequence
        class_out = [classes[2][0]]
        return class_out

    def _get_filenames_and_classes(self, raw_folder, process_folder):
        # Get list from parent class
        list_of_filenames, list_of_corresponding_class_names, list_of_unique_class_names, list_of_grouping_data = super(Dataset, self)._get_filenames_and_classes(raw_folder, process_folder)
        
        # Filter list based on names
        print('Filtering examples...')
        list_of_filenames_new = []
        list_of_corresponding_class_names_new = []
        list_of_grouping_data_new = []
        # Loop throguh all examples
        for filename, corresponding_class, grouping in zip(list_of_filenames, list_of_corresponding_class_names, list_of_grouping_data):
            basename_without_extension = os.path.splitext(os.path.basename(filename))[0]
            basename_parts = basename_without_extension.split('_')
            if (basename_parts[6][1] == '0') and (basename_parts[7] == '1'):
                # Add example to new lists
                list_of_filenames_new.append(filename)
                list_of_corresponding_class_names_new.append(corresponding_class)
                list_of_grouping_data_new.append(grouping)
            else:
                pass
        
        print('Unfiltered examples: ' , len(list_of_filenames))
        print('Filtered examples  : ' , len(list_of_filenames_new))

        return list_of_filenames_new, list_of_corresponding_class_names_new, list_of_unique_class_names, list_of_grouping_data_new
