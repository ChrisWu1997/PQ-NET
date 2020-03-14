from config.config_pqnet import PQNetConfig
from config.config_lgan import LatentGANConfig


def get_config(name):
    if name == 'pqnet':
        return PQNetConfig
    elif name == 'lgan':
        return LatentGANConfig
    else:
        raise ValueError("Got name: {}".format(name))
