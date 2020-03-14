import torch
import torch.nn as nn
from networks import get_network
from agent.base import BaseAgent
from util.visualization import project_voxel_along_xyz, visualize_sdf


class PartAEAgent(BaseAgent):
    def __init__(self, config):
        super(PartAEAgent, self).__init__(config)
        self.points_batch_size = config.points_batch_size
        self.resolution = config.resolution
        self.batch_size = config.batch_size

    def build_net(self, config):
        net = get_network('part_ae', config)
        print('-----Part AE architecture-----')
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()

    def forward(self, data):
        input_vox3d = data['vox3d'].cuda()  # (shape_batch_size, 1, dim, dim, dim)
        points = data['points'].cuda()  # (shape_batch_size, points_batch_size, 3)
        target_sdf = data['values'].cuda()  # (shape_batch_size, points_batch_size, 1)

        output_sdf = self.net(points, input_vox3d)

        loss = self.criterion(output_sdf, target_sdf)
        return output_sdf, {"mse": loss}

    def visualize_batch(self, data, mode, outputs=None):
        tb = self.train_tb if mode == 'train' else self.val_tb

        parts_voxel = data['vox3d'][0][0].numpy()
        data_points64 = data['points'][0].numpy() * self.resolution
        data_values64 = data['values'][0].numpy()
        output_sdf = outputs[0].detach().cpu().numpy()

        target = visualize_sdf(data_points64, data_values64, concat=True, vox_dim=self.resolution)
        output = visualize_sdf(data_points64, output_sdf, concat=True, vox_dim=self.resolution)
        voxel_proj = project_voxel_along_xyz(parts_voxel, concat=True)
        tb.add_image("voxel", torch.from_numpy(voxel_proj), self.clock.step, dataformats='HW')
        tb.add_image("target", torch.from_numpy(target), self.clock.step, dataformats='HW')
        tb.add_image("output", torch.from_numpy(output), self.clock.step, dataformats='HW')
