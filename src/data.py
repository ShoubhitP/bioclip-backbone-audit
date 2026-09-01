from torchvision.datasets import Flowers102
from torch.utils.data import ConcatDataset

def load_flowers102(root, transform=None):
    train_dataset = Flowers102(root=root, split='train', download=True, transform=transform)
    val_dataset = Flowers102(root=root, split='val', download=True, transform=transform)
    test_dataset = Flowers102(root=root, split='test', download=True, transform=transform)

    support_pool = ConcatDataset([train_dataset, val_dataset])
    return support_pool, test_dataset