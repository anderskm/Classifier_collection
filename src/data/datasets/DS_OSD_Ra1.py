import src.data.datasets.DS_Seeds as superDataset
import numpy as np
import PIL
import tensorflow as tf
import os
import glob

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD Randers Lot1',
                    rawFolder = 'data/raw/OSD_Ra1_eq',
                    processFolder = 'data/processed/OSD_Ra1_eq',
                    numShards = 10):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)

    def _get_filenames_and_classes(self, raw_folder, process_folder):

        data_root = os.path.join(raw_folder)

        list_of_filenames = []
        list_of_corresponding_class_names = []
        list_of_grouping_data = []

        list_of_lab_directories = [f.path for f in os.scandir(data_root) if f.is_dir()]
        for lab_directory in list_of_lab_directories:

            lab_name = os.path.basename(lab_directory)

            list_of_species_directories = [f.path for f in os.scandir(lab_directory) if f.is_dir()]
            for species_directory in list_of_species_directories:

                species_name = os.path.basename(species_directory)
                
                list_of_lot_directories = glob.glob(os.path.join(species_directory, 'Lot*'))
                for lot_directory in list_of_lot_directories:
                    list_of_filenames_in_lot = glob.glob(os.path.join(lot_directory,'*.tif'))
                    list_of_filenames += list_of_filenames_in_lot
                    list_of_corresponding_class_names += [[species_name] for i in range(len(list_of_filenames_in_lot))]
                    list_of_grouping_data += [lab_name for i in range(len(list_of_filenames_in_lot))]

                # TODO: Loop through weed folders. Set species_name to weed
                #  list_of_lot_directories = glob.glob(os.path.join(species_directory, 'Lot*'))

        list_of_corresponding_class_names_T = list(map(list, zip(*list_of_corresponding_class_names))) # Transpose list of lists
        list_of_unique_class_names = [list(set(class_names)) for class_names in list_of_corresponding_class_names_T]

        return list_of_filenames, list_of_corresponding_class_names, list_of_unique_class_names, list_of_grouping_data
