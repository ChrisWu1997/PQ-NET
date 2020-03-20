import torch
import torch.nn as nn

# Part AutoEncoder based on ImNet
##############################################################################

class Encoder3D(nn.Module):
    def __init__(self, n_layers, ef_dim=32, z_dim=128):
        super(Encoder3D, self).__init__()
        model = []

        in_channels = 1
        out_channels = ef_dim
        for i in range(n_layers - 1):
            model.append(nn.Conv3d(in_channels, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1,
                                   bias=False))  # FIXME: in IM-NET implementation, they use bias before BN
            model.append(nn.BatchNorm3d(num_features=out_channels, momentum=0.1))  # FIXME: momentum value
            model.append(nn.LeakyReLU(0.02))

            in_channels = out_channels
            out_channels *= 2

        model.append(nn.Conv3d(out_channels // 2, z_dim, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=0))
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out


class ImDecoderSkipConnect(nn.Module):
    def __init__(self, n_layers, f_dim, z_dim):
        """With skip connection"""
        super(ImDecoderSkipConnect, self).__init__()
        in_channels = z_dim + 3
        out_channels = f_dim * (2 ** (n_layers - 2))
        model = []
        for i in range(n_layers - 1):
            if i > 0:
                in_channels += z_dim + 3
            if i < 4:
                model.append([nn.Linear(in_channels, out_channels), nn.Dropout(p=0.4), nn.LeakyReLU()])
            else:
                model.append([nn.Linear(in_channels, out_channels), nn.LeakyReLU()])
            in_channels = out_channels
            out_channels = out_channels // 2
        model.append([nn.Linear(in_channels, 1), nn.Sigmoid()])

        self.layer1 = nn.Sequential(*model[0])
        self.layer2 = nn.Sequential(*model[1])
        self.layer3 = nn.Sequential(*model[2])
        self.layer4 = nn.Sequential(*model[3])
        self.layer5 = nn.Sequential(*model[4])
        self.layer6 = nn.Sequential(*model[5])

    def forward(self, points, z):
        z_cat = torch.cat([points, z], dim=1)
        out = self.layer1(z_cat)
        out = self.layer2(torch.cat([out, z_cat], dim=1))
        out = self.layer3(torch.cat([out, z_cat], dim=1))
        out = self.layer4(torch.cat([out, z_cat], dim=1))
        out = self.layer5(torch.cat([out, z_cat], dim=1))
        out = self.layer6(out)
        return out


class PartImNetAE(nn.Module):
    def __init__(self, en_n_layers, ef_dim, de_n_layers, df_dim, z_dim):
        super(PartImNetAE, self).__init__()
        self.encoder = Encoder3D(en_n_layers, ef_dim, z_dim)
        assert de_n_layers == 6
        self.decoder = ImDecoderSkipConnect(de_n_layers, df_dim, z_dim)

    def forward(self, points, vox3d):
        """

        :param points: (shape_batch_size, point_batch_size, 3)
        :param vox3d: (shape_batch_size, 1, dim, dim, dim)
        :return: predicted per-part sdf (shape_batch_size, point_batch_size, 1)
        """
        shape_batch_size = points.size(0)
        point_batch_size = points.size(1)
        z = self.encoder(vox3d)  # (shape_batch_size, z_dim)
        # z = torch.cat([z, affine], dim=1)
        batch_z = z.unsqueeze(1).repeat((1, point_batch_size, 1)).view(-1, z.size(1))
        batch_points = points.view(-1, 3)

        out = self.decoder(batch_points, batch_z)
        out = out.view((shape_batch_size, point_batch_size, -1))
        return out


def test():
    pass


if __name__ == '__main__':
    test()
