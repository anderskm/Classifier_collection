import src.data.datasets.DS_OSD as superDataset

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, seed only, new focus',
                    rawFolder = 'data/raw/extracted_crops_new_focus__masked__sanitized',
                    processFolder = 'data/processed/OSD_seed_new_focus',
                    numShards = 162):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
