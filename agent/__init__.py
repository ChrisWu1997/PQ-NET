from agent.agent_partae import PartAEAgent
from agent.agent_seq2seq import Seq2SeqAgent
from agent.agent_lgan import WGANAgant
from agent.pqnet import PQNET


def get_agent(config):
    if config.module == 'part_ae':
        return PartAEAgent(config)
    elif config.module == 'seq2seq':
        return Seq2SeqAgent(config)
    else:
        raise ValueError

