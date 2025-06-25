from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
import math



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 输出的中间通道数
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def RN_backbone(pretrained=True, progress=True,model_path=None, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class TransLayer(nn.Module):
    def __init__(self, block):
        super(TransLayer, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 1024
        self.groups = 1
        self.base_width = 64
        self.layer4 = self._make_layer(block, 512, stride=2)

    def _make_layer(self, block, planes, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        return out


class Trans_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(Trans_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            if _ == 1 and self.is_local:
                layers.append(nn.Identity())
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def RN_refine(is_local=True, pretrained=True, progress=True,model_path=None, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)
    return model


def Trans(pretrained=True,model_path=None):
    model = TransLayer(Bottleneck)
    if pretrained:
        state_dict = torch.load(model_path)
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.0' in name:
                # print(name)
                pretrain_keys.append(name)
        for key in pretrain_keys:
            model.state_dict()[key].copy_(state_dict[key])
        # model.load_state_dict(state_dict, strict=False)
    # print(model.state_dict())
    return model


def Trans_refine(is_local=True, pretrained=True, progress=True,model_path=None, **kwargs):
    model = Trans_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load(model_path)
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.1' in name or 'layer4.2' in name:
                # if 'layer4' in name:
                pretrain_keys.append(name)
        for key in pretrain_keys:
            key2 = list(key)
            if int(key2[7]) == 1:
                key2[7] = '{}'.format(int(key2[7]) - 1)
            # key2[7] = '{}'.format(int(key2[7]) - 1)
            key2 = ''.join(key2)
            model.state_dict()[key2].copy_(state_dict[key])
    # print(model.state_dict())
    return model

class ImageMaskPredictor(nn.Module):

    def __init__(self, embed_dim=384):
        super().__init__()

        # 输出分支
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim ,128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1),  #
        )

    def forward(self, x, ratio=0.5):

        # combined = torch.cat([feat, global_feat], dim=1)  # (B, C + C//2, H, W)

        #
        mask = self.out_conv(x)  # (B, 2, H, W)

        return mask

class GraphConv(nn.Module):
    def __init__(self, feature_dim, topk):
        super(GraphConv, self).__init__()

        self.topk = topk
        self.nn = nn.Sequential(
            nn.Conv2d(feature_dim * topk, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

    def forward(self, x, mask=None):  # x: B, C, H, W mask: B, 1, H, W
        features_self = x
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # B,HW,C

        if mask is not None:
            mask = mask.view(B, 1, H * W).permute(0, 2, 1)  # B,HW,1
            x_norm = F.normalize(x,p=2, dim=-1)  # B,HW,C
            x_mask = x_norm*mask  # B, N, C
            cos_sim_mask = x_mask.bmm(x_mask.permute(0, 2, 1).contiguous())  #
        else:
            x_norm = F.normalize(x,p=2, dim=-1)  # B,HW,C
            cos_sim_mask = x_norm.bmm(x_norm.permute(0, 2, 1).contiguous())  #

        topk_weight, topk_index = torch.topk(cos_sim_mask, k=self.topk+1, dim=-1)
        topk_weight = topk_weight[:, :, 1:]  #
        topk_index = topk_index[:, :, 1:]  #

        topk_index = topk_index.to(torch.long)

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(B).view(-1, 1, 1).to(topk_index.device)  #
        selected_features = x[batch_indices, topk_index, :]  #

        topk_weight = F.softmax(topk_weight, dim=2)  #
        #
        x_graph = torch.mul(topk_weight.unsqueeze(-1), selected_features) # B,HW, K, C

        B, N, K, C = x_graph.shape
        x_graph = x_graph.view(B, N, -1) # B,N,K*C
        x_graph = x_graph.permute(0, 2, 1) # B,K*C,N
        x_graph = x_graph.reshape(B,-1,H,W) # B,K*C,H,W
        x_graph = self.nn(x_graph) # B,C,H,W

        if mask is not None:
            mask = mask.permute(0, 2, 1).reshape(B, 1, H, W)  # B,1,H,W
            x_out = x_graph*mask + features_self
        else:
            x_out = x_graph+features_self

        return x_out

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, topk=2):
        super(Grapher, self).__init__()
        self.channels = in_channels

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = GraphConv(in_channels, topk=topk)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  #

    def forward(self, x, mask=None):

        x = self.fc1(x)
        x = self.graph_conv(x,mask)
        x = self.fc2(x)

        return x


class MDH(nn.Module):
    def __init__(self, code_length=12, num_classes=200, feat_size=2048, device='cpu', pretrained=False):
        super(MDH, self).__init__()
        # 骨干网
        model_path = '/data/yhx/pretrain_model/resnet/resnet50.pth'
        self.backbone = RN_backbone(pretrained=pretrained, model_path=model_path)

        self.trans = Trans(pretrained=pretrained, model_path=model_path)
        self.mask_predictor = ImageMaskPredictor(embed_dim=2048)

        self.conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.cos_graph = Grapher(in_channels=256, topk=2)
        self.conv2 = nn.Conv2d(256, 2048, kernel_size=1)

        self.refine_global = RN_refine(is_local=False, pretrained=pretrained, model_path=model_path)
        self.refine_local = Trans_refine(pretrained=pretrained, model_path=model_path)

        self.cls = nn.Linear(feat_size, num_classes)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length

        # global
        if self.code_length != 32:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2  , feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        else:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2 , feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))

        # local
        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))

        #
        self.gate = nn.Sequential(
            nn.Linear(2 * 2048, 2),  #
            nn.Softmax(dim=1)
        )

        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def get_param_groups(self):
        param_groups = [[], []]  # backbone, otherwise
        for name, param in super().named_parameters():
            if "backbone" in name:
                param_groups[0].append(param)
            else:
                param_groups[1].append(param)
        return param_groups

    def forward(self, x):
        out = self.backbone(x)  # .detach()
        global_f = self.refine_global(out)

        local_f = self.trans(out)

        mask_logits = self.mask_predictor(local_f) # B,2,H,W
        mask = F.softmax(mask_logits, dim=1)[:, 0:1, :, :]  #
        fore_feature = local_f * mask
        back_feature = local_f * (1.0 - mask)

        fore_feature = self.conv1(fore_feature)
        back_feature = self.conv1(back_feature)
        if self.training:
            mask_binary = F.gumbel_softmax(mask_logits, hard=True, dim=1)[:, 0:1, :, :]  #
        else:
            mask_binary = torch.argmin(mask_logits, dim=1, keepdim=True)  # (B, 1, H, W)
        fore_feature = self.cos_graph(fore_feature, mask_binary)
        back_feature = self.cos_graph(back_feature, 1.0 - mask_binary)
        fore_feature = self.conv2(fore_feature)
        back_feature = self.conv2(back_feature)
        # back_feature = self.cos_graph(back_feature)

        # local_f1: B,2048,7,7  avg_local_f1: B,2048
        fore_feature1, avg_fore_feature = self.refine_local(fore_feature)
        back_feature2, avg_back_feature = self.refine_local(back_feature)

        combined = torch.cat([avg_fore_feature, avg_back_feature], dim=1)
        weights = self.gate(combined)  # 形状：[B, 2]
        avg_feature = weights[:, 0:1] * avg_fore_feature + weights[:, 1:2] * avg_back_feature
        # avg_feature = (avg_fore_feature + avg_back_feature)

        deep_S_G = F.linear(global_f, self.W_G)

        deep_S_1 = F.linear(avg_feature, self.W_L1)
        # deep_S_2 = F.linear(enhance_feature, self.W_L2)

        #
        deep_S = torch.cat([deep_S_G, deep_S_1], dim=1)
        #
        ret = self.hash_layer_active(deep_S)

        ##########

        # if self.training:
        decision_mask = mask_binary
        fore_weight = weights[:, 0:1]

        if self.training:
            cls = self.cls(global_f)
            cls1 = self.cls(avg_feature)
            # cls2 = self.cls(avg_fore_feature)
            # cls3 = self.cls_loc(avg_local_f3)
            return ret, cls, cls1, fore_weight, decision_mask
        return ret


def mdh(code_length=12, num_classes=200, feat_size=2048, device='cpu', pretrained=False, **kwargs):
    # att_size = 1
    model = MDH(code_length, num_classes, feat_size, device, pretrained, **kwargs)
    return model
