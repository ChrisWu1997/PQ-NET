from networks.networks_partae import PartImNetAE
from networks.networks_seq2seq import Seq2SeqAE
from networks.networks_lgan import Generator, Discriminator
from dataset.data_utils import n_parts_map


def get_network(name, config):
    if name == 'part_ae':
        net = PartImNetAE(config.en_n_layers, config.en_f_dim, config.de_n_layers, config.de_f_dim, config.en_z_dim)
    elif name == 'seq2seq':
        part_feat_size = config.en_z_dim + config.boxparam_size
        en_input_size = part_feat_size + n_parts_map(config.max_n_parts) + 1
        de_input_size = part_feat_size
        net = Seq2SeqAE(en_input_size, de_input_size, config.hidden_size)
    elif name == 'G':
        net = Generator(config.n_dim, config.h_dim, config.z_dim)
    elif name == 'D':
        net = Discriminator(config.h_dim, config.z_dim)
    else:
        raise ValueError
    return net


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
