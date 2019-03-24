import random
import torch.utils.data as data
from glob import glob
from utils import load_images, data_augmentation


# folder structure
# data
#   |train
#   |  |low
#   |  |  |*.png
#   |  |high
#   |  |  |*.png
#   |eval
#   |  |low
#   |  |  |*.png
def get_dataset_len(route, phase):
    if phase == 'train':
        low_data_names = glob(route + phase + '/low/*.png')
        high_data_names = glob(route + phase + '/high/*.png')
        low_data_names.sort()
        high_data_names.sort()
        assert len(low_data_names) == len(high_data_names)
        return len(low_data_names), [low_data_names, high_data_names]
    elif phase == 'eval':
        low_data_names = glob(route + phase + '/low/*.png')
        return len(low_data_names), low_data_names
    else:
        return 0, []


def getitem(phase, data_names, item, patch_size):
    if phase == 'train':
        low_im = load_images(data_names[0][item])
        high_im = load_images(data_names[1][item])

        h, w, _ = low_im.shape
        x = random.randint(0, h - patch_size)
        y = random.randint(0, w - patch_size)
        rand_mode = random.randint(0, 7)

        low_im = data_augmentation(low_im[x:x + patch_size, y:y + patch_size, :], rand_mode)
        high_im = data_augmentation(high_im[x:x + patch_size, y:y + patch_size, :], rand_mode)

        return low_im, high_im
    elif phase == 'eval':
        low_im = load_images(data_names[item])
        return low_im


class TheDataset(data.dataset):

    def __init__(self, route='./data/', phase='train', patch_size=48):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.len, self.data_names = get_dataset_len(route, phase)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return getitem(self.phase, self.data_names, item, self.patch_size)
