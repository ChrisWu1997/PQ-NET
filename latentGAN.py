import os
import numpy as np
import h5py
from torch.utils.data import DataLoader
from config import get_config
from dataset import ShapeCodesDataset
from agent import WGANAgant
from util.utils import ensure_dir


def main():
    # create experiment config
    config = get_config('lgan')()

    # create network and training agent
    tr_agent = WGANAgant(config)

    if config.is_train:
        # load from checkpoint if provided
        if config.cont:
            tr_agent.load_ckpt(config.ckpt)

        # create dataloader
        dataset = ShapeCodesDataset(config.data_root)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, worker_init_fn=np.random.seed(), drop_last=True)

        tr_agent.train(train_loader)
    else:
        # load trained weights
        tr_agent.load_ckpt(config.ckpt)

        # run generator
        generated_shape_codes = tr_agent.generate(config.n_samples)

        # save generated z
        save_path = os.path.join(config.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(config.ckpt, config.n_samples))
        ensure_dir(os.path.dirname(save_path))
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset("zs", shape=generated_shape_codes.shape, data=generated_shape_codes)


if __name__ == '__main__':
    main()