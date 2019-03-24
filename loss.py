import torch
import torch.nn.functional as F
import torchvision


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
    input_r = torchvision.transforms.functional.to_grayscale(input_r)


if __name__ == '__main__':
    tensor = torch.rand(1, 300, 400, 1)
    out_data = ave_gradient(tensor, 'x')
    print(out_data.size())
