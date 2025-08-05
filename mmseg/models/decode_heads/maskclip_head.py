# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MaskClipHead(BaseDecodeHead):

    def __init__(self, text_categories, text_channels, text_embeddings_path,
                 visual_projs_path, vit=False, ks_thresh=0., pd_thresh=0.,
                 attn_pooling=False, num_heads=32, **kwargs):
        super(MaskClipHead, self).__init__(**kwargs)

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.visual_projs_path = visual_projs_path

        # register text_embeddings as buffer (NOT trainable)
        # These will be used only for initializing the classifier weights
        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))
        self.load_text_embeddings()

        # Define trainable classifier
        # This will replace F.conv2d(..., text_embeddings) and be optimized during training
        self.classifier = nn.Conv2d(
            in_channels=text_channels,
            out_channels=text_categories,
            kernel_size=1,
            bias=False
        )

        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.k_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.c_proj = nn.Conv2d(self.in_channels, text_channels, 1)

        self.load_visual_projs()
        self.init_classifier_with_text_embeddings()

        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh
        self.attn_pooling = attn_pooling
        self.num_heads = num_heads

    def init_weights(self):
        super(MaskClipHead, self).init_weights()
        self.load_text_embeddings()
        self.load_visual_projs()
        self.init_classifier_with_text_embeddings()
        
        # Explicitly ensure classifier is trainable (default is True, but clarified here)
        for param in self.classifier.parameters():
            param.requires_grad = True

        # Freeze all CLIP visual projection layers (they remain unchanged during training)
        freeze_layers = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for name in freeze_layers:
            m = getattr(self, name)
            for p in m.parameters():
                p.requires_grad = False

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings.copy_(loaded)
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def init_classifier_with_text_embeddings(self):
        # Initialize classifier weights using normalized CLIP text embeddings
        # This provides semantic priors to start training from a meaningful point
        with torch.no_grad():
            normed_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=1, keepdim=True)
            self.classifier.weight.copy_(normed_embeddings[:, :, None, None])

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded proj weights from {self.visual_projs_path}', logger=get_root_logger())

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        q, k, v, cls_token = None, None, None, None

        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            feat = self.proj(v) if v is not None else self.proj(x)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            if self.attn_pooling:
                N, C, H, W = x.shape
                x = x.view(N, C, -1).permute(2, 0, 1)
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
                x, _ = F.multi_head_attention_forward(
                    query=x, key=x, value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight[:, :, 0, 0],
                    k_proj_weight=self.k_proj.weight[:, :, 0, 0],
                    v_proj_weight=self.v_proj.weight[:, :, 0, 0],
                    in_proj_weight=None,
                    in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight[:, :, 0, 0],
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False
                )
                feat = x[1:].permute(1, 2, 0).view(N, -1, H, W)
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                q = torch.flatten(q, start_dim=2).transpose(-2, -1)
                k = torch.flatten(k, start_dim=2).transpose(-2, -1)
                v = self.v_proj(x)
                feat = self.c_proj(v)

        output = self.cls_seg(feat)
        if not self.training:
            output = self.refine_output(output, k)
        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = self.classifier(feat)  # classifier is now trainable
        return output

    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output * 100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output * 100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # Standard training logic using trainable classifier
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
