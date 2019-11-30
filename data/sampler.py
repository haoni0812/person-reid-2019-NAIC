import random
import collections
from torch.utils.data import sampler

import numpy as np
import torch

class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id
#        self.top_N = 1000

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.imgs):
            _id = data_source.id(path)
            self._id2index[_id].append(idx)
            
        self.weights = self.class_balanced(self._id2index)
        print(self.weights)
            


    def __iter__(self):
#        unique_ids = self.my_unique_ids( self._id2index, self.batch_id)
#        print(len(unique_ids))  
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        


        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)


    def __len__(self):
        return len(self.data_source.unique_ids) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
    
    
    #put the ids which have more than 200 images in a single batch, a represents orginal unique_id list
    def my_unique_ids(self, _id2index, batch_id):
        all_id = []
        for single_id in self.data_source.unique_ids:
            k = int(len(_id2index[single_id])/4) if len(_id2index[single_id])>4 else 1
            all_id.extend([single_id]*k)
            
        random.shuffle(all_id)
        print(len(all_id))
        
        my_unique_ids = []
        batch = []
        for my_id in all_id:
            batch.append(my_id)
            if len(set(batch)) == batch_id:
                my_unique_ids.extend(list(set(batch)))
                batch = []

        return my_unique_ids
        
        
    #compute class balanced weights
    def class_balanced(self, id2index):
        beta = 0.99

        samples_per_cls = []
        for i in id2index.keys():
            samples_per_cls.append(len(id2index[i]))
        print(len(id2index),len(samples_per_cls))

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        print(effective_num)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(id2index)
        weights = torch.tensor(weights).float()
        
        return weights
    
        
        
