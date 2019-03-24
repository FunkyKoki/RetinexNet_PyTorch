import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient(input_tensor, direction):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    h, w = input_tensor.size()[2], input_tensor.size()[3]

    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y

    out = F.conv2d(input_tensor, kernel, padding=(1, 1))
    out = torch.abs(out[:, :, 0:h, 0:w])

    return out.permute(0, 2, 3, 1)


def ave_gradient(input_tensor, direction):
    return (F.avg_pool2d(gradient(input_tensor, direction).permute(0, 3, 1, 2), 3, stride=1, padding=1))\
        .permute(0, 2, 3, 1)


def smooth(input_l, input_r):
    rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140])
    input_r = torch.tensordot(input_r, rgb_weights, dims=([-1], [-1]))
    input_r = torch.unsqueeze(input_r, -1)

    return torch.mean(
        gradient(input_l, 'x') * torch.exp(-10 * ave_gradient(input_r, 'x')) +
        gradient(input_l, 'y') * torch.exp(-10 * ave_gradient(input_r, 'y'))
    )


class DecomLoss(nn.Module):

    def __init__(self):
        super(DecomLoss, self).__init__()

    def forward(self, r_low, l_low, r_high, l_high, input_low, input_high):
        l_low_3 = torch.cat((l_low, l_low, l_low), -1)
        l_high_3 = torch.cat((l_high, l_high, l_high), -1)

        recon_loss_low = torch.mean(torch.abs(r_low * l_low_3 - input_low))
        recon_loss_high = torch.mean(torch.abs(r_high * l_high_3 - input_high))
        recon_loss_mutal_low = torch.mean(torch.abs(r_high * l_low_3 - input_low))
        recon_loss_mutal_high = torch.mean(torch.abs(r_low * l_high_3 - input_high))
        equal_r_loss = torch.mean(torch.abs(r_low - r_high))

        ismooth_loss_low = smooth(l_low, r_low)
        ismooth_loss_high = smooth(l_high, r_high)

        return \
            recon_loss_low + recon_loss_high +\
            0.001*recon_loss_mutal_low + 0.001*recon_loss_mutal_high + \
            0.1*ismooth_loss_low + 0.1*ismooth_loss_high + \
            0.01*equal_r_loss


class RelightLoss(nn.Module):

    def __init__(self):
        super(RelightLoss, self).__init__()

    def forward(self, l_delta, r_low, input_high):
        l_delta_3 = torch.cat((l_delta, l_delta, l_delta), -1)

        relight_loss = torch.mean(torch.abs(r_low * l_delta_3 - input_high))

        ismooth_loss_delta = smooth(l_delta, r_low)

        return relight_loss + 3 * ismooth_loss_delta


if __name__ == '__main__':
    tensor = torch.rand(1, 300, 400, 1)
    out_data = smooth(tensor, torch.rand(1, 300, 400, 3))
    print(out_data)
