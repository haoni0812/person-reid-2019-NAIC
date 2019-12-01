import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck, resnet101

def make_model(args):
    return MGN(args)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

class MGN(nn.Module):
    def __init__(self, args):
        super(MGN, self).__init__()
        num_classes = args.num_classes

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
#        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes)
#        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes)
#        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes)
#
#        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes)
#        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes)
#        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes)
#        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes)
#        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes)
#
#        self._init_fc(self.fc_id_2048_0)
#        self._init_fc(self.fc_id_2048_1)
#        self._init_fc(self.fc_id_2048_2)
#
#        self._init_fc(self.fc_id_256_1_0)
#        self._init_fc(self.fc_id_256_1_1)
#        self._init_fc(self.fc_id_256_2_0)
#        self._init_fc(self.fc_id_256_2_1)
#        self._init_fc(self.fc_id_256_2_2)
        
        self.bn_0 = nn.BatchNorm1d(args.feats)
        self.bn_1 = nn.BatchNorm1d(args.feats)
        self.bn_2 = nn.BatchNorm1d(args.feats)
        
        self.bn_1_0 = nn.BatchNorm1d(args.feats)
        self.bn_1_1 = nn.BatchNorm1d(args.feats)
        self.bn_2_0 = nn.BatchNorm1d(args.feats)
        self.bn_2_1 = nn.BatchNorm1d(args.feats)
        self.bn_2_2 = nn.BatchNorm1d(args.feats)
        
        self.bn_0.bias.requires_grad_(False)
        self.bn_1.bias.requires_grad_(False)
        self.bn_2.bias.requires_grad_(False)

        self.bn_1_0.bias.requires_grad_(False)
        self.bn_1_1.bias.requires_grad_(False)
        self.bn_2_0.bias.requires_grad_(False)
        self.bn_2_1.bias.requires_grad_(False)
        self.bn_2_2.bias.requires_grad_(False)


        self.fc_id_2048_0 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_2048_1 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_2048_2 = nn.Linear(args.feats, num_classes, bias=False)

        self.fc_id_256_1_0 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_256_1_1 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_256_2_0 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_256_2_1 = nn.Linear(args.feats, num_classes, bias=False)
        self.fc_id_256_2_2 = nn.Linear(args.feats, num_classes, bias=False)

        self.bn_0.apply(weights_init_kaiming)
        self.bn_1.apply(weights_init_kaiming)
        self.bn_2.apply(weights_init_kaiming)

        self.bn_1_0.apply(weights_init_kaiming)
        self.bn_1_1.apply(weights_init_kaiming)
        self.bn_2_0.apply(weights_init_kaiming)
        self.bn_2_1.apply(weights_init_kaiming)
        self.bn_2_2.apply(weights_init_kaiming)

        self.fc_id_2048_0.apply(weights_init_classifier)
        self.fc_id_2048_1.apply(weights_init_classifier)
        self.fc_id_2048_2.apply(weights_init_classifier)
        
        self.fc_id_256_1_0.apply(weights_init_classifier)
        self.fc_id_256_1_1.apply(weights_init_classifier)
        self.fc_id_256_2_0.apply(weights_init_classifier)
        self.fc_id_256_2_1.apply(weights_init_classifier)
        self.fc_id_256_2_2.apply(weights_init_classifier)




    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        
#        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
#        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
#        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        
        fg_p1_bn = self.bn_0(fg_p1)
        fg_p2_bn = self.bn_0(fg_p2)
        fg_p3_bn = self.bn_0(fg_p3)
        
        f0_p2_bn = self.bn_0(f0_p2)
        f1_p2_bn = self.bn_0(f1_p2)
        f0_p3_bn = self.bn_0(f0_p3)
        f1_p3_bn = self.bn_0(f1_p3)
        f2_p3_bn = self.bn_0(f2_p3)
        
        l_p1 = self.fc_id_2048_0(fg_p1_bn)
        l_p2 = self.fc_id_2048_1(fg_p2_bn)
        l_p3 = self.fc_id_2048_2(fg_p3_bn)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2_bn)
        l1_p2 = self.fc_id_256_1_1(f1_p2_bn)
        l0_p3 = self.fc_id_256_2_0(f0_p3_bn)
        l1_p3 = self.fc_id_256_2_1(f1_p3_bn)
        l2_p3 = self.fc_id_256_2_2(f2_p3_bn)
        
        #put the feature on the hyper ball surface

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

        


