from torchvision.datasets import ImageFolder

class ImagePathLoader(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path = self.imgs[index][0]
        return sample, target, path