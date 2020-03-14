import torch
import numpy as np
import os
from outside_code.libmise import MISE
from networks import get_network
from util.visualization import partsdf2mesh, partsdf2voxel, affine2bboxes


class PQNET(object):
    def __init__(self, config):
        self.points_batch_size = config.points_batch_size
        self.boxparam_size = config.boxparam_size
        self.vox_dim = 64
        self.threshold = config.threshold
        self.upsampling_steps = config.upsampling_steps

        self.resolution = self.vox_dim * (1 << self.upsampling_steps)

        self.load_network(config)

        torch.random.initial_seed()

    def load_network(self, config):
        """load trained network module: seq2seq and part_ae"""
        self.seq2seq = get_network("seq2seq", config)
        name = config.ckpt if config.ckpt == 'latest' else "ckpt_epoch{}".format(config.ckpt)
        seq2seq_model_path = os.path.join(config.model_dir, "{}.pth".format(name))
        self.seq2seq.load_state_dict(torch.load(seq2seq_model_path)['model_state_dict'])
        print("Load Seq2Seq model from: {}".format(seq2seq_model_path))
        self.seq2seq = self.seq2seq.cuda().eval()

        part_imnet = get_network("part_ae", config)
        part_imnet.load_state_dict(torch.load(config.partae_modelpath)['model_state_dict'])
        print("Load PartAE model from: {}".format(config.partae_modelpath))
        self.part_encoder = part_imnet.encoder.cuda().eval()
        self.part_decoder = part_imnet.decoder.cuda().eval()

    def infer_part_encoder(self, parts_voxel):
        """run part ae encoder to map part voxels to vectors

        :param parts_voxel:  (n_parts, 1, vox_dim, vox_dim, vox_dim)
        :return: part_codes: (n_parts, en_z_dim)
        """
        part_codes = self.part_encoder(parts_voxel)  # (n_parts, en_z_dim)
        return part_codes

    def infer_part_decoder(self, part_codes, points):
        """run part ae decoder to calculate part sdf

        :param part_codes: (n_parts, 1, en_z_dim)
        :param points: (n_parts, n_points, 3) value range (0, 1)
        :return: out: ndarray (n_parts, n_points, 1) output sdf values for each point
        """
        pred_n_parts = part_codes.size(0)
        if points.size(0) != pred_n_parts:
            raise RuntimeError("pred:{} gt:{}".format(pred_n_parts, points.size(0)))
        n_points = points.size(1)

        num = n_points // self.points_batch_size
        if n_points % self.points_batch_size > 0:
            num += 1
        output_sdf = []
        for i in range(num):
            batch_points = points[:, i * self.points_batch_size:(i + 1) * self.points_batch_size, :]
            cur_n_points = batch_points.size(1)
            batch_z = part_codes.repeat((1, cur_n_points, 1)).view(-1, part_codes.size(-1))
            batch_points = batch_points.contiguous().view(-1, 3)

            out = self.part_decoder(batch_points, batch_z)
            out = out.view((pred_n_parts, cur_n_points, -1))
            out = out.detach().cpu().numpy()
            output_sdf.append(out)
        output_sdf = np.concatenate(output_sdf, axis=1)
        return output_sdf

    def set_data(self, data):
        """set data as inputs"""
        parts_voxel = data['vox3d'].cuda()
        batch_size, max_n_parts, vox_dim = parts_voxel.size(0), parts_voxel.size(1), parts_voxel.size(-1)
        parts_voxel = parts_voxel.view(-1, 1, vox_dim, vox_dim, vox_dim)
        affine = data['affine_input'].cuda()
        cond = data['cond'].cuda()
        self.input_n_parts = data['n_parts'][0]

        part_codes = self.part_encoder(parts_voxel)  # (n_parts, en_z_dim)
        part_codes = part_codes.view(batch_size, max_n_parts, -1).transpose(0, 1)
        cond_pack = cond.unsqueeze(0).repeat(affine.size(0), 1, 1)
        self.input_seq = torch.cat([part_codes, affine, cond_pack], dim=2) # .unsqueeze(1)

    def encode_seq(self, input_seq):
        """run seq2seq encoder to encode input part sequence"""
        hidden = self.seq2seq.infer_encoder(input_seq)
        return hidden

    def decode_seq(self, hidden_code, length=None):
        """run seq2seq decoder to decode part sequence"""
        output_seq, output_stop = self.seq2seq.infer_decoder_stop(hidden_code, length)

        pred_n_parts = output_seq.size(0)
        self.output_part_codes = output_seq[:, :, :-self.boxparam_size].detach()
        self.output_affine = output_seq[:, :, -self.boxparam_size:].detach().cpu().numpy()
        self.n_parts = pred_n_parts

    def reconstruct_seq(self, input_seq):
        """run seq2seq to reconstruct input sequence"""
        self.decode_seq(self.encode_seq(input_seq), length=input_seq.size(0))

    def reconstruct(self, data):
        """reconstruct input data"""
        self.set_data(data)
        self.decode_seq(self.encode_seq(self.input_seq), length=self.input_seq.size(0))

    def encode(self, data):
        """encode input data to shape latent space"""
        self.set_data(data)
        return self.encode_seq(self.input_seq)

    def eval_part_points(self, points, part_idx):
        """eval sdf values of each part

        :param points: (n_points, 3) value range (0, 1)
        :param part_idx: int
        :return:
            output sdf values: (n_points,)
        """
        values = self.infer_part_decoder(self.output_part_codes[part_idx:part_idx+1], points.unsqueeze(0))
        return values.squeeze()

    def transform_points(self, points, values):
        """transform part points from local frame to global frame

        :param points: (n_parts, n_points, 3) or [(n_points1, 3), (n_points2, 3), ...], in range (0, self.vox_dim)
        :param values: (n_parts, n_points, 1) or [(n_points1, 1), (n_points2, 1), ...]
        :return:
        """
        cube_mid = np.asarray([self.resolution // 2, self.resolution // 2, self.resolution // 2]).reshape(1, 3)
        new_points, new_values = [], []
        for idx in range(len(points)):
            part_points = points[idx]
            part_values = values[idx]
            part_translation = self.output_affine[idx, 0, :3].reshape(1, 3) * self.resolution
            part_size = self.output_affine[idx, 0, 3:6].reshape(1, 3)

            part_scale = np.max(part_size)
            part_points = (part_points - cube_mid) * part_scale + part_translation

            mins = part_translation - part_size * self.resolution / 2
            maxs = part_translation + part_size * self.resolution / 2
            in_bbox_indice = np.max(part_points - maxs, axis=1)
            in_bbox_indice = np.where(in_bbox_indice <= 0)[0]
            part_points = part_points[in_bbox_indice, :]
            part_values = part_values[in_bbox_indice]

            in_bbox_indice = np.max(part_points - mins, axis=1)
            in_bbox_indice = np.where(in_bbox_indice >= 0)[0]
            part_points = part_points[in_bbox_indice, :]
            part_values = part_values[in_bbox_indice]

            part_points = np.clip(part_points, 0, self.resolution - 1)
            new_points.append(part_points)
            new_values.append(part_values)
        return new_points, new_values

    def eval_part_sdf(self, part_idx):
        """get output part sdf

        :param part_idx: int
        :return: all_points: (n_points, 3)
                 all_values: (n_points, )
        """
        mesh_extractor = MISE(self.vox_dim, self.upsampling_steps, self.threshold)

        points = mesh_extractor.query()

        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).cuda()
            # rescale to range (0, 1)
            pointsf = pointsf / mesh_extractor.resolution

            values = self.eval_part_points(pointsf, part_idx).astype(np.double)

            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        all_points, all_values = mesh_extractor.get_points()
        return all_points, all_values

    def generate_shape(self, format='voxel', by_part=True):
        """generate final shape geometry

        :param format: str. output geometry format
        :param by_part: bool. segment each part or put as a whole
        :return:
        """
        points = []
        values = []
        for idx in range(self.n_parts):
            part_points, part_values = self.eval_part_sdf(idx)
            points.append(part_points)
            values.append(part_values)
        points, values = self.transform_points(points, values)
        if format == 'sdf':
            return (points, values)
        elif format == 'voxel':
            shape_voxel = partsdf2voxel(points, values, vox_dim=self.resolution, by_part=by_part)
            return shape_voxel
        elif format == 'mesh':
            shape_mesh = partsdf2mesh(points, values, affine=None, vox_dim=self.resolution, by_part=by_part)
            return shape_mesh
        else:
            raise NotImplementedError
