from .dataset_example import BraTS_new, ISLES_TrainDataset

datasets = {
    'new': BraTS_new,
    'isles2022': ISLES_TrainDataset
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)