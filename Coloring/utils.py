import torch
import torchvision
import torchvision.transforms as transforms

class ColorAndGray(object):
    def __call__(self, img):
        # ToTensor()の前に呼ぶ場合はimgはPILのインスタンス
        gray = img.convert("L")
        return img, gray

# 複数の入力をtransformsに展開するラッパークラスを作る
class MultiInputWrapper(object):
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        else:
            return [self.base_func(x) for x in xs]


def load_datasets():
    transform = transforms.Compose([
        ColorAndGray(),
        MultiInputWrapper(transforms.ToTensor()),
        MultiInputWrapper([
            transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,)),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    ])
    trainset = torchvision.datasets.STL10(root="./data",
                                          split="unlabeled",
                                          download=True,
                                          transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=4, pin_memory=True)
    return train_loader


