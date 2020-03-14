from torch.utils.data import DataLoader
from dataset.dataset_partae import PartAEDataset
from dataset.dataset_seq2seq import Seq2SeqDataset, pad_collate_fn_for_dict
from dataset.dataset_lgan import ShapeCodesDataset
from dataset.data_utils import load_from_hdf5_by_part
import numpy as np


def get_dataloader(phase, config, use_all_points=False, is_shuffle=None):
    is_shuffle = phase == 'train' if is_shuffle is None else is_shuffle

    if config.module == 'part_ae':
        dataset = PartAEDataset(phase, config.data_root, config.category, config.points_batch_size,
                                all_points=use_all_points, resolution=config.resolution)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers, worker_init_fn=np.random.seed())
    elif config.module == 'seq2seq':
        dataset = Seq2SeqDataset(phase, config.data_root, config.category, config.max_n_parts)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers, collate_fn=pad_collate_fn_for_dict)
    else:
        raise NotImplementedError
    return dataloader
