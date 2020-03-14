import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from util.visualization import draw_parts_bbox_voxel
from networks import get_network, set_requires_grad
from agent.base import BaseAgent


class Seq2SeqAgent(BaseAgent):
    def __init__(self, config):
        super(Seq2SeqAgent, self).__init__(config)
        self.stop_weight = config.stop_weight
        self.boxparam_size = config.boxparam_size
        self.teacher_decay = config.teacher_decay
        self.teacher_forcing_ratio = 0.5

        self.bce_min = torch.tensor(1e-3, dtype=torch.float32, requires_grad=True).cuda()

    def build_net(self, config):
        # restore part encoder
        part_imnet = get_network('part_ae', config)
        if not os.path.exists(config.partae_modelpath):
            raise ValueError("Pre-trained part_ae path not exists: {}".format(config.partae_modelpath))
        part_imnet.load_state_dict(torch.load(config.partae_modelpath)['model_state_dict'])
        print("Load pre-trained part AE from: {}".format(config.partae_modelpath))
        self.part_encoder = part_imnet.encoder.cuda().eval()
        self.part_decoder = part_imnet.decoder.cuda().eval()
        set_requires_grad(self.part_encoder, requires_grad=False)
        set_requires_grad(self.part_decoder, requires_grad=False)
        del part_imnet
        # build rnn
        net = get_network('seq2seq', config).cuda()
        return net

    def set_optimizer(self, config):
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

    def set_loss_function(self):
        self.rec_criterion = nn.MSELoss(reduction='none').cuda()
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

    def update_teacher_forcing_ratio(self):
        self.teacher_forcing_ratio *= self.teacher_decay

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if self.clock.epoch < 2000:
            self.scheduler.step(self.clock.epoch)

    def forward(self, data):
        row_vox3d = data['vox3d']   # (B, T, 1, vox_dim, vox_dim, vox_dim)
        batch_size, max_n_parts, vox_dim = row_vox3d.size(0), row_vox3d.size(1), row_vox3d.size(-1)
        batch_n_parts = data['n_parts']
        target_stop = data['sign'].cuda()
        bce_mask = data['mask'].cuda()
        affine_input = data['affine_input'].cuda()
        affine_target = data['affine_target'].cuda()
        cond = data['cond'].cuda()

        batch_vox3d = row_vox3d.view(-1, 1, vox_dim, vox_dim, vox_dim).cuda()
        with torch.no_grad():
            part_geo_features = self.part_encoder(batch_vox3d)   # (B * T, z_dim)
            part_geo_features = part_geo_features.view(batch_size, max_n_parts, -1).transpose(0, 1)
            cond_pack = cond.unsqueeze(0).repeat(affine_input.size(0), 1, 1)

            target_part_geo = part_geo_features.detach()
            part_feature_seq = torch.cat([part_geo_features, affine_input, cond_pack], dim=2)
            part_feature_seq = pack_padded_sequence(part_feature_seq, batch_n_parts, enforce_sorted=False)
            _, seq_lengths = pad_packed_sequence(part_feature_seq)  # self to self translation
            target_seq = torch.cat([target_part_geo, affine_target], dim=2)

        output_seq, output_stop = self.net(part_feature_seq, target_seq, self.teacher_forcing_ratio)

        bce_loss = self.bce_criterion(output_stop, target_stop) * bce_mask * self.stop_weight

        code_rec_loss = self.rec_criterion(output_seq[:, :, :-self.boxparam_size], target_part_geo) * bce_mask
        param_rec_loss = self.rec_criterion(output_seq[:, :, -self.boxparam_size:], affine_target) * bce_mask

        code_rec_loss = torch.sum(code_rec_loss) / (torch.sum(bce_mask) * code_rec_loss.size(2))
        param_rec_loss = torch.sum(param_rec_loss) / (torch.sum(bce_mask) * param_rec_loss.size(2))
        bce_loss = torch.max(torch.sum(bce_loss) / torch.sum(bce_mask), self.bce_min)

        return output_seq, {"code": code_rec_loss, "param": param_rec_loss, "stop": bce_loss}

    def visualize_batch(self, data, mode, outputs=None, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        n_parts = data['n_parts'][0]
        affine_input = data['affine_input'][:n_parts, 0].detach().cpu().numpy()
        affine_target = data['affine_target'][:n_parts, 0].detach().cpu().numpy()
        affine_output = outputs[:n_parts, 0, -self.boxparam_size:].detach().cpu().numpy()

        # draw box
        bbox_proj = draw_parts_bbox_voxel(affine_input)
        tb.add_image("bbox_input", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')
        bbox_proj = draw_parts_bbox_voxel(affine_target)
        tb.add_image("bbox_target", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')
        bbox_proj = draw_parts_bbox_voxel(affine_output)
        tb.add_image("bbox_output", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')
