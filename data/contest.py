from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
import os

#author:watsom
class contest(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader
        
        self._id2label_ = {}
        self.imgs = []

        data_path = args.datadir
        if dtype == 'train':
            with open(data_path + '/train_list.txt', 'r', encoding='utf-8') as f1:
                lables = f1.readlines()
            for i,each in enumerate(lables):
                if i>=1 and i<=len(lables)-2:
                    k=0
                    for j in range(3):
                        neborline = lables[i-1+j].split(' ')
                        if each.split(' ')[-1] == neborline[-1]:
                            k=k+1
                    if k>1:
                        self._id2label_[each.split(' ')[0].split('/')[-1]] = each.split(' ')[-1]
                        self.imgs.append(os.path.join(data_path+'/train_set',each.split(' ')[0].split('/')[-1]))
                else:
                    self._id2label_[each.split(' ')[0].split('/')[-1]] = each.split(' ')[-1]
                    self.imgs.append(os.path.join(data_path+'/train_set',each.split(' ')[0].split('/')[-1]))
            self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
            print(len(self._id2label))
            print(len(self.imgs))

        elif dtype == 'test':
            data_path += '/gallery_a'
            self._id2label_ = {each: each.split('.')[0] for each in os.listdir(data_path)}
            self.imgs = [path for path in list_pictures(data_path)]
            self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        else:
            data_path += '/query_a'
            self._id2label_ = {each: each.split('.')[0] for each in os.listdir(data_path)}
            self.imgs = [path for path in list_pictures(data_path)]
            self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target,path

    def __len__(self):
        return len(self.imgs)

    def id(self,file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(self._id2label_[file_path.split('/',8)[-1]])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return list(set(self.ids))

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return 1

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """

        return [1 for path in self.imgs]