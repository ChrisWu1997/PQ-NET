import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, n_dim, h_dim, z_dim):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(n_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, z_dim),
        )
        self.main = main
        self.apply(weights_init)

    def forward(self, noise):
        # if FIXED_GENERATOR:
        #     return noise + real_data
        # else:
        output = self.main(noise)
        output = torch.tanh(output)
        return output


class Discriminator(nn.Module):

    def __init__(self, h_dim, z_dim):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(h_dim, 1),
        )
        self.main = main
        self.apply(weights_init)

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
