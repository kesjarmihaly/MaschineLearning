import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Subset

from skimage.measure import compare_ssim, compare_psnr

from PIL import Image

class LRandHR(object):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def __call__(self, img):
        # ToTensor()の前に呼ぶ場合はimgはPILのインスタンス
        w, h = img.size
        input = img.resize((w//self.upscale_factor, h//self.upscale_factor), Image.NEAREST)
        return input, img

# 複数の入力をtransformsに展開するラッパークラスを作る
class MultiInputWrapper(object):
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        else:
            return [self.base_func(x) for x in xs]

def split_train_and_test(dataset):
    n_samples = len(dataset)
    train_size = round(n_samples * 0.7)

    subset1_indices = list(range(0, train_size))
    subset2_indices = list(range(train_size, n_samples))

    train_dataset = Subset(dataset, subset1_indices)
    test_dataset = Subset(dataset, subset2_indices)

    return train_dataset, test_dataset


def load_datasets(up_scale_factor, batch_size):
    transform = transforms.Compose([
        LRandHR(up_scale_factor),
        MultiInputWrapper(transforms.ToTensor()),
        MultiInputWrapper([
            transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,)),
            transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,))
        ])
    ])
    dataset = torchvision.datasets.STL10(root="./data",
                                          split="unlabeled",
                                          download=True,
                                          transform=transform)
    trainset, testset = split_train_and_test(dataset)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader( testset, batch_size=batch_size,
                                               shuffle=False, pin_memory=True)
    return train_loader, test_loader

def ssim(img1, img2):
    val = compare_ssim(img1, img2, multichannel=True)
    return val

def psnr(img1, img2):
    val = compare_psnr(img1, img2, data_range=1)
    return val

