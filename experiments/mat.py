import torch.utils.data as data
import scipy.io as sio
import numpy as np
from PIL import Image
import os
import os.path

MAT_EXTENSIONS = [
    '.mat', '.MAT'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in MAT_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_loader(path):
    return sio.loadmat(path)['Y1'].astype('float32')


class MatFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError(
                "Found 0 images in subfolders of: " + root + "\n"
                "Supported image extensions are: " + ",".join(MAT_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# train_file=MatFolder(r"C:\Users\l\Desktop\备份代码\20230829代码\batchspd\batchspd\experiments\data\afew\train")
# val_file=MatFolder(r"C:\Users\l\Desktop\备份代码\20230829代码\batchspd\batchspd\experiments\data\afew\val")
# train_first_elements = [item[0] for item in train_file]
# trX = np.stack(train_first_elements, axis=0)
#
# train_second_elements = [item[1] for item in train_file]
# trY = np.stack(train_second_elements, axis=0)
#
#
# val_first_elements = [item[0] for item in val_file]
# teX = np.stack(val_first_elements, axis=0)
#
# val_second_elements = [item[1] for item in val_file]
# teY = np.stack(val_second_elements, axis=0)
#
# np.savez("data/afew_dataset.npz",trX=trX,trY=trY,teX=teX,teY=teY)
