import numpy as np
import timm
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from methods.zoomnet.mlp import INR
from utils.builder import MODELS
from utils.ops import cus_sample


class SIU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s_pre_up(s)
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(ResidualBlock, self).__init__()
        hidden_dim = int(inp // expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self,
                 inp_dim,
                 out_dim
                 ):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        # self.theta_phi = InvertedResidualBlock(inp=inp_dim, oup=inp_dim, expand_ratio=2)
        # self.theta_rho = InvertedResidualBlock(inp=inp_dim, oup=out_dim, expand_ratio=2)
        self.weight_encoder = nn.Sequential(
            ResidualBlock(inp=inp_dim, oup=inp_dim // 2, expand_ratio=2),
            ResidualBlock(inp=inp_dim // 2, oup=out_dim, expand_ratio=2)
        )
        self.bottlenect = nn.Sequential(
            InvertedResidualBlock(inp=inp_dim, oup=inp_dim, expand_ratio=2),
            InvertedResidualBlock(inp=inp_dim, oup=inp_dim, expand_ratio=2)
        )
        # self.theta_eta = InvertedResidualBlock(inp=inp_dim, oup=out_dim, expand_ratio=2)

    def forward(self, xr, xt):
        w_xr = self.weight_encoder(xr)
        w_xt = self.weight_encoder(xt)

        gamma_xr = xr + self.bottlenect(xr)
        gamma_xt = xt + self.bottlenect(xt)

        dynamic = torch.cat([w_xr, w_xt], dim=1)

        return dynamic, gamma_xr, gamma_xt


def save_weight(tensor_data, time):
    # 将Tensor转换为字符串格式，这里使用join来连接行，使用'\t'作为分隔符
    data_string = '\n'.join(['\t'.join(map(str, row)) for row in tensor_data])

    # 定义输出文件的路径
    output_file_path = 'tensor_output.txt'

    # 将字符串写入到文件中
    with open(output_file_path, 'a') as file:
        file.write(time + '\n' + data_string)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, 2 * n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            # nn.BatchNorm2d(num_groups * hidden_dim),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = rearrange(x, 'b c h w -> b (h w) c')
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x= self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=384, w=384)
        return x

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


@MODELS.register()
class CMMFSwin(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.INR_train = False
        dim = 128
        encoder1 = timm.create_model(model_name="convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, in_chans=3)

        self.encoder_shared_level1 = nn.Sequential(encoder1.stem_0, encoder1.stem_1, encoder1.stages_0)
        self.encoder_shared_level2 = nn.Sequential(encoder1.stages_1)
        self.encoder_rgb_private_level3 = encoder1.stages_2
        self.encoder_rgb_private_level4 = encoder1.stages_3
        encoder2 = timm.create_model(model_name="convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, in_chans=3)
                                     # pretrained_cfg_overlay=dict(file='D:\\Yang\\model_pretrain\\model.safetensors'))
        self.encoder_thermal_private_level3 = encoder2.stages_2
        self.encoder_thermal_private_level4 = encoder2.stages_3

        self.loss_Func = nn.L1Loss()
        for i in range(4):
            setattr(self, f'fusion_stage{i}', DetailNode((dim * (2 ** i)), 1))
            setattr(self, f'INR{i}', INR(2).cuda())

        self.d1 = nn.Sequential(HMU((dim * (2 ** 0)), num_groups=4, hidden_dim=dim // 2))
        self.upsample_level3 = Upsample(dim * (2 ** 1))
        self.d2 = nn.Sequential(HMU((dim * (2 ** 1)), num_groups=4, hidden_dim=dim))
        self.upsample_level2 = Upsample(dim * (2 ** 2))
        self.d3 = nn.Sequential(HMU((dim * (2 ** 2)), num_groups=4, hidden_dim=dim * 2))
        self.upsample_level1 = Upsample(dim * (2 ** 3))
        self.d4 = nn.Sequential(HMU((dim * (2 ** 3)), num_groups=4, hidden_dim=dim * 2))
        self.up = FinalPatchExpand_X4(input_resolution=(96, 96), dim_scale=4, dim=dim)
        self.output = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, bias=False)

    def encoder_translayer(self, x):
        # en_feats = self.shared_encoder(x)
        en_feats = []
        f1 = self.encoder_shared_level1(x)
        f2 = self.encoder_shared_level2(f1)
        f3 = self.encoder_rgb_private_level3(f2)
        f4 = self.encoder_rgb_private_level4(f3)
        en_feats.append(f1)
        en_feats.append(f2)
        en_feats.append(f3)
        en_feats.append(f4)

        return en_feats

    def t_encoder_translayer(self, x):
        # en_feats = self.thermal_encoder(x)

        en_feats = []
        f1 = self.encoder_shared_level1(x)
        f2 = self.encoder_shared_level2(f1)
        f3 = self.encoder_rgb_private_level3(f2)
        f4 = self.encoder_rgb_private_level4(f3)

        en_feats.append(f1)
        en_feats.append(f2)
        en_feats.append(f3)
        en_feats.append(f4)

        return en_feats

    def body(self, m_scale, thermal):

        feats = self.encoder_translayer(m_scale)
        t_feats = self.t_encoder_translayer(thermal)

        fused_list = []
        loss_NR = torch.Tensor([0.0]).cuda()
        for i in range(len(feats)):
            xr = feats[i]
            xt = t_feats[i]
            gate = getattr(self, f'fusion_stage{i}')
            dynamic, xr, xt = (gate(xr, xt))

            if self.INR_train:
                INR_Trans = getattr(self, f'INR{i}')
                dynamic_NR = INR_Trans(dynamic)
                loss_NR += self.loss_Func(dynamic, dynamic_NR)
                dynamic_NR = dynamic_NR.chunk(2, dim=1)
                dynamic_xr = (torch.abs(dynamic_NR[0]) + 1e-30) / (
                        torch.abs(dynamic_NR[0]) + torch.abs(dynamic_NR[1]) + 1e-30)
                dynamic_xt = (torch.abs(dynamic_NR[1]) + 1e-30) / (
                        torch.abs(dynamic_NR[0]) + torch.abs(dynamic_NR[1]) + 1e-30)
            else:
                dynamic = dynamic.chunk(2, dim=1)
                dynamic_xr = (torch.abs(dynamic[0]) + 1e-30) / (
                    torch.abs(dynamic[0]) + torch.abs(dynamic[1]) + 1e-30)
                dynamic_xt = (torch.abs(dynamic[1]) + 1e-30) / (
                    torch.abs(dynamic[0]) + torch.abs(dynamic[1]) + 1e-30)
            fused = xr * dynamic_xr + xt * dynamic_xt
            fused_list.append(fused)

        feats = fused_list
        feats.reverse()

        x = self.d4(feats[0])
        x = self.upsample_level1(x)

        x = self.d3(x + feats[1])
        x = self.upsample_level2(x)

        x = self.d2(x + feats[2])
        x = self.upsample_level3(x)

        x = self.d1(x + feats[3])
        x = self.up(x)
        logits = self.output(x)
        return dict(seg=logits), loss_NR

    def train_forward(self, data, **kwargs):
        # assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)

        output, loss_NR = self.body(
            m_scale=data["image1.0"],
            thermal=data["thermal"]
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
            loss_NR=loss_NR
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output, loss = self.body(
            m_scale=data["image1.0"],
            thermal=data["thermal"]
        )
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, loss_NR, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        losses.append(loss_NR)
        loss_str.append(f"loss_NR: {loss_NR.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            # if name.startswith("shared_encoder.layer"):
            #     param_groups.setdefault("pretrained", []).append(param)
            # elif name.startswith("encoder"):
            #     param_groups.setdefault("pretrained", []).append(param)
            # elif name.startswith("shared_encoder."):
            #     param_groups.setdefault("fixed", []).append(param)
            # else:
            param_groups.setdefault("retrained", []).append(param)
        return param_groups

    def get_grouped_INR_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("d"):
                param_groups.setdefault("retrained", []).append(param)
            elif name.startswith("INR"):
                param_groups.setdefault("retrained", []).append(param)
            elif name.startswith("logits."):
                param_groups.setdefault("retrained", []).append(param)
            else:
                param_groups.setdefault("fixd", []).append(param)
        return param_groups

if __name__ == '__main__':
    img_rgb, img_thermal = torch.randn(1, 3, 384, 384), torch.randn(1, 3, 384, 384)
    img_rgb = img_rgb.cuda()
    img_thermal = img_thermal.cuda()
    model = CMMFSwin().cuda()
    input = {"image1.0": img_rgb, "thermal": img_thermal}
    model.test_forward(input)
    print('a')
