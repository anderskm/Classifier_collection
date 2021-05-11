import src.data.datasets.DS_OSD as superDataset

class Dataset(superDataset.Dataset):
    
    def __init__(self,
                    name = 'OSD, seed only, new white clover focus',
                    rawFolder = 'data/raw/OSD_seed_new_wc',
                    processFolder = 'data/processed/OSD_seed_new_wc',
                    numShards = 162):
        super(Dataset, self).__init__(name = name, rawFolder = rawFolder, processFolder = processFolder, numShards = numShards)
