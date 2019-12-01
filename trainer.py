import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
import json

class Trainer():
    def __init__(self, args, model, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer_39.pt'))
            )
            model.load(r'E:\XianjunLuo\accepted\person_reid_match\experiment\test') #鍔犺浇瑕佸湪鏂偣澶勭户缁缁冪殑妯″瀷锛屼唬鐮佷粠model鏂囦欢涓媉init_.py璋冪敤
            self.scheduler.last_epoch -= 1
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()


    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch+1
        print(epoch)
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, labels,path) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            


            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')

        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        print(epoch)
#        self.ckpt.write_log('\n[INFO] Test:')
#        self.model.eval()

#        self.ckpt.add_log(torch.zeros(1, 5))
#        qf = self.extract_feature(self.query_loader)[0].numpy()
#        gf = self.extract_feature(self.test_loader)[0].numpy()

        query_imgs = self.extract_feature(self.query_loader)[1]
        gallery_imgs = self.extract_feature(self.test_loader)[1]

        if self.args.re_rank:
            print('re-ranking')
#            q_g_dist = np.dot(qf, np.transpose(gf))
#            q_q_dist = np.dot(qf, np.transpose(qf))
#            g_g_dist = np.dot(gf, np.transpose(gf))
#            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
#        else:
#            dist = cdist(qf, gf)
        sub = {}
        top=200
        dista = np.load('TestB_distance_4768_offi_base_ibn101a_woRE_rerank.npy')
        print(dista.shape)
        distb = np.load('TestB_distance_4768_one_ibn101a_bdb_rerank.npy')
        print(distb.shape)
        distc = np.load('TestB_distance_4768_one_ibn101a_rankedloss_rerank.npy')
        print(distc.shape)
        dist = dista + distb + distc
        for i in range(dist.shape[0]):
            query2galley = []
            reid_list = dist[i].argsort()
            reid_list = reid_list[::-1]
            for j in range(top):
                query2galley.append(gallery_imgs[reid_list[j]])
                sub[query_imgs[i].split("/")[-1]] = [a.split("/")[-1] for a in query2galley]
            with open('.\\submission_{}.json'.format(str('ensemble3')), 'w', encoding='utf-8') as f1:
                f1.write(json.dumps(sub))
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))
        if epoch == 0:
            np.save("dist.npy",dist)
#        '''single shot – 指gallery中每个人的图像为一张（N=1）；
#            muti shot – 指gallery中每个人的图像为多张（N>1），同样的Rank-1下，一般N越大，得到的识别率越高。'''
#        '''r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
#                separate_camera_set=False,
#                single_gallery_shot=False,
#                first_match_break=)'''

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        imgs_path = []
        for (inputs, labels,path) in loader:
            imgs_path.extend(each.split('\\',8)[-1] for each in path)
#            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
#            for i in range(2):
#                if i==1:
#                    inputs = self.fliphor(inputs)
#                input_img = inputs.to(self.device)
#                outputs = self.model(input_img)
#                f = outputs[0].data.cpu()
#                ff = ff + f

#            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
#            ff = ff.div(fnorm.expand_as(ff))

#            features = torch.cat((features, ff), 0)
            features = 0

        return features,imgs_path

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch > self.args.epochs
