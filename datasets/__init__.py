import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset
from .pl_pair_dock_guide_dataset import PocketLigandPairDockGuideDataset
from .pdbbind import PDBBindDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    elif name == 'pdbbind':
        dataset = PDBBindDataset(root, *args, **kwargs)
    elif name == 'pl_dock_guide':
        dataset = PocketLigandPairDockGuideDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
