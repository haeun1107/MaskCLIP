# mmseg/models/decode_heads/maskclip_plus_head.py
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from mmseg.ops import resize
from ..builder import HEADS, build_head, build_backbone
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MaskClipPlusHead(BaseDecodeHead):

    def __init__(self, decode_module_cfg, text_categories, text_channels, 
                    text_embeddings_path, cls_bg=False, norm_feat=False, 
                    start_self_train=(-1, -1), start_clip_guided=(-1, -1), 
                    unlabeled_cats=[], clip_unlabeled_cats=[], clip_cfg=None,
                    clip_weights_path=None, reset_counter=False, clip_channels=None, 
                    vit=False, ks_thresh=0., pd_thresh=0., conf_thresh=0., 
                    distill=False, distill_labeled=True, distill_weight=1., **kwargs):
        super(MaskClipPlusHead, self).__init__(
            input_transform=decode_module_cfg.pop('input_transform'), **kwargs)
        # Text embedding info: number of classes and embedding dimensions
        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        # Whether to L2 normalize feature before classification
        self.norm_feat = norm_feat
        # Class indices for which pseudo-labels will be generated
        self.unlabeled_cats = torch.tensor(unlabeled_cats, dtype=torch.long, device='cuda')
        self.clip_unlabeled_cats = torch.tensor(clip_unlabeled_cats, dtype=torch.long, device='cuda')
        # Training iteration settings for pseudo-label switching
        self.start_self_train = start_self_train
        self.start_clip_guided = start_clip_guided
        self.self_train = (start_self_train[0] >= 0)
        self.clip_guided = (start_clip_guided[0] >= 0)
        self.train_unlabeled = self.self_train or self.clip_guided
        # Iteration counter for tracking training progress
        self.register_buffer('_iter_counter', torch.tensor(0, device='cuda'))
        self.clip_weights_path = clip_weights_path
        self.cls_bg = cls_bg
        self.reset_counter = reset_counter
        if clip_channels is None:
            clip_channels = self.in_channels
        self.distill = distill
        self.distill_labeled = distill_labeled
        self.distill_weight = distill_weight

        del self.conv_seg
        self.init_cfg = None

        decode_module_cfg.update(kwargs)
        self.build_decode_module(decode_module_cfg)

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))
        
        # Add trainable classifier
        self.classifier = nn.Conv2d(
            in_channels=self.channels,
            out_channels=text_categories, 
            kernel_size=1,
            bias=False
        )
                
        self.vit = vit
        # Build CLIP backbone if CLIP-guided labeling is enabled
        if self.clip_guided:
            self.clip = build_backbone(clip_cfg)
            self.ks_thresh = ks_thresh
            self.pd_thresh = pd_thresh
            self.conf_thresh = conf_thresh
            if vit:
                self.proj = nn.Conv2d(clip_channels, text_channels, 1, bias=False)
            else:
                self.q_proj = nn.Conv2d(clip_channels, clip_channels, 1)
                self.k_proj = nn.Conv2d(clip_channels, clip_channels, 1)
                self.v_proj = nn.Conv2d(clip_channels, clip_channels, 1)
                self.c_proj = nn.Conv2d(clip_channels, text_channels, 1)
        # Whether to learn a background embedding vector (for cls_bg=True)
        if cls_bg:
            self.bg_embeddings = nn.Parameter(torch.randn(1, text_channels))
        self.clip_up_unlabeled_cats = []

    def init_weights(self, call_super=True):
        if call_super:
            super(MaskClipPlusHead, self).init_weights()
        self.load_text_embeddings()
            
        if self.clip_guided:
            self.load_clip_weights()
            
        self.init_classifier_with_text_embeddings()
        
        for p in self.classifier.parameters():
            p.requires_grad = True

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def init_classifier_with_text_embeddings(self):
        # Initialize classifier weights from normalized text embeddings
        with torch.no_grad():
            normed = self.text_embeddings / self.text_embeddings.norm(dim=1, keepdim=True)
            self.classifier.weight.copy_(normed[:, :, None, None])

    def load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.clip.load_state_dict(loaded['clip'])
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _freeze(self):
        """Freeze params and norm stats."""
        super(MaskClipPlusHead, self)._freeze()
        # always freeze these modules
        if self.clip_guided:
            attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
            attrs.append('clip')
            for attr in attrs:
                i = getattr(self, attr)
                for m in i.modules():
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        # never freeze bg_classifier
        if self.cls_bg:
            self.bg_embeddings.requires_grad = True
    
    def build_decode_module(self, cfg):
        cfg['init_cfg'] = None
        cfg['in_channels'] = self.in_channels
        cfg['channels'] = self.channels
        self.decode_module = build_head(cfg)
        del self.decode_module.loss_decode
        del self.decode_module.conv_seg
        del self.decode_module.dropout

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        if self.norm_feat:
            feat = feat / feat.norm(dim=1, keepdim=True)
        #output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        # Classify pixels using trainable classifier layer
        output = self.classifier(feat)
        
        if self.cls_bg:
            bg_weight = self.bg_embeddings / self.bg_embeddings.norm(dim=-1, keepdim=True)
            bg = F.conv2d(feat, bg_weight[:, :, None, None])
            output = torch.cat([bg, output], dim=1)
        
        return output

    def forward(self, inputs):
        # Apply segmentation decode module (e.g. DeepLabV2) to input features
        output = self.decode_module.forward_module(inputs)

        # Detach for pseudo-label generation (no gradient)
        feat = output.detach()
        # Pixel-wise classification with cosine similarity to text embeddings
        output = self.cls_seg(output)

        if self.reset_counter:
            self.reset_counter = False
            self._iter_counter *= 0

        # Handle training iteration counter and logs
        self._iter_counter += 1
        if self.training:
            if self._iter_counter == self.start_self_train[0]:
                print_log('Start self training', logger=get_root_logger())
            if self._iter_counter == self.start_self_train[1]:
                print_log('Stop self training', logger=get_root_logger())
            if self._iter_counter == self.start_clip_guided[0]:
                print_log('Start clip guided training', logger=get_root_logger())
            if self._iter_counter == self.start_clip_guided[1]:
                print_log('Stop clip guided training', logger=get_root_logger())

        # If in training and using pseudo-labels, return both logit and feature
        if self.train_unlabeled:
            return [output, feat]
        #print(f"[DEBUG] Iteration: {self._iter_counter.item()} | Training: {self.training}")
        return [output]


    def assign_label(self, gt_semantic_seg, feat, norm=False, unlabeled_cats=None,
                        clip=False, k=None, cls_token=None):
        # If gt is a tuple, unpack it (from CLIP-guided loss output)
        if isinstance(gt_semantic_seg, tuple):
            #] gt_semantic_seg is a tuple. Unpacking...")
            gt_clip, loss_clip, acc_clip = gt_semantic_seg
            gt_semantic_seg = gt_clip

        # Skip if no unlabeled pixels exist
        if (gt_semantic_seg < 0).sum() == 0:
            print("[DEBUG] All pixels are labeled. assign_label skipped.")
            return gt_semantic_seg, None
        
        # Normalize feature for cosine similarity if enabled
        if norm:
            feat = feat / feat.norm(dim=1, keepdim=True)

        gt_semantic_seg = gt_semantic_seg.squeeze(1)

        # Combine background and foreground embeddings if cls_bg is True
        if self.cls_bg:
            bg_embeddings = self.bg_embeddings / self.bg_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = torch.cat([bg_embeddings, self.text_embeddings], dim=0)
        else:
            text_embeddings = self.text_embeddings
        #print("[DEBUG] feat stats:", feat.mean().item(), feat.std().item())
        #print("[DEBUG] text_embeddings stats:", self.text_embeddings.mean().item())

        # Select embeddings corresponding to unlabeled classes
        # [unlabeled_cats, text_channels]
        unlabeled_text = text_embeddings[unlabeled_cats]
        unlabeled_idx = (gt_semantic_seg < 0) # select -1 (unlabeled classes)
        
        # Compute cosine similarity between pixel features and text embeddings
        output = torch.einsum('nchw,lc->nlhw', [feat, unlabeled_text])
        #print(f"[DEBUG] einsum output stats → min: {output.min()}, max: {output.max()}, mean: {output.mean()}")
        if clip:
            output = self.refine_clip_output(output, k)
        
        # Resize similarity map to match segmentation mask resolution
        output = resize(
            input=output,
            size=gt_semantic_seg.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        # print(f"[DEBUG] assign_label() called with unlabeled_cats: {unlabeled_cats}")
        # print(f"[DEBUG] output shape: {output.shape}")
        # print(f"[DEBUG] gt_semantic_seg unique: {torch.unique(gt_semantic_seg)}")

        neg_pos = None
        if self.conf_thresh > 0:
            N, C, H, W = output.shape
            neg_pos = output.view(N, C, -1).max(dim=1)[0] < self.conf_thresh
            neg_pos = neg_pos.view(N, H, W)
        
        # Select best matching class for each pixel
        output = output.permute(0, 2, 3, 1)
        #print(f"[DEBUG] unlabeled_idx count: {(unlabeled_idx).sum().item()}")
        if output.shape[1] == 0:
            print("⚠️ CLIP-guided label output is empty! Skipping label assignment.")
            return None
        match_matrix = output[unlabeled_idx]
        preds = match_matrix.argmax(dim=1)
        #print(f"[DEBUG] match_matrix.argmax label distribution: {torch.bincount(preds)}")

        # Assign predicted class index to unlabeled pixels
        gt_semantic_seg[unlabeled_idx] = unlabeled_cats[match_matrix.argmax(dim=1)]
        if neg_pos is not None:
            gt_semantic_seg[neg_pos] = -1

        return gt_semantic_seg[:, None, :, :]

    def label_sanity_check(self, gt_semantic_seg):
        for i in self.unlabeled_cats:
            if not torch.all(gt_semantic_seg != i):
                print(f"[❌] Ground-truth leakage! class {i} found in unlabeled_cats")
                print("    unique values in GT:", torch.unique(gt_semantic_seg))
                raise AssertionError(f'Ground-truth leakage! {i}')
                
        for i in self.clip_up_unlabeled_cats:
            if not torch.all(gt_semantic_seg != i):
                print(f"[❌] Ground-truth leakage! class {i} found in clip_up_unlabeled_cats")
                print("    unique values in GT:", torch.unique(gt_semantic_seg))
                raise AssertionError(f'Ground-truth leakage! {i}')

    def refine_clip_output(self, output, k=None):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # Handle case where gt_semantic_seg is a tuple (e.g. from CLIP-guided evaluation)
        if isinstance(gt_semantic_seg, tuple):
            gt_clip, loss_clip, acc_clip = gt_semantic_seg
            gt_semantic_seg = gt_clip 

        if self.distill:
            seg_logits, feat = self.forward(inputs)
            # Get CLIP feature map from image
            x = self.clip(img)[-1]
            # Extract visual features from CLIP for ViT or ResNet
            if self.vit:
                v = None
                if isinstance(x, list) and len(x) == 4:
                    x, _, _, v = x
                if isinstance(x, list) and len(x) == 2:
                    x, cls_token = x
                if v is not None:
                    clip_feat = self.proj(v)
                else:
                    clip_feat = self.proj(x)
            else:
                clip_feat = self.c_proj(self.v_proj(x))
            # Normalize CLIP features
            clip_feat = clip_feat / clip_feat.norm(dim=1, keepdim=True)

            # Select pixels for distillation loss: labeled or unlabeled
            if self.distill_labeled:
                mask = (gt_semantic_seg != 255)
            else:
                mask = (gt_semantic_seg < 0)

            # Make sure mask is properly handled
            if isinstance(gt_semantic_seg, tuple):
                #print("[DEBUG] gt_semantic_seg is a tuple. Unpacking in decode_head...")
                gt_clip, loss_clip, acc_clip = gt_semantic_seg
                gt_clip[gt_clip < 0] = 255
                gt_semantic_seg = gt_clip
            else:
                gt_semantic_seg[gt_semantic_seg < 0] = 255

            # Compute supervised segmentation loss
            losses = self.losses(seg_logits, gt_semantic_seg)

            # Resize both features to match original image size
            feat = resize(
                input=feat,
                size=mask.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            clip_feat = resize(
                input=clip_feat,
                size=mask.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            
            # Compute distillation loss on valid pixels only
            mask = mask.squeeze(1)
            if torch.any(mask):
                feat = feat.permute(0, 2, 3, 1)[mask]
                clip_feat = clip_feat.permute(0, 2, 3, 1)[mask]
                losses['loss_distill'] = F.l1_loss(feat, clip_feat) * self.distill_weight
        
        # UNLABELED TRAINING (Self / CLIP Guided)
        elif self.train_unlabeled:
            seg_logits, feat = self.forward(inputs)
            # gt_self, gt_clip, gt_weight = None, None, None
            
            # # Make sure mask is properly handled
            # if isinstance(gt_semantic_seg, tuple):
            #     #print("[DEBUG] gt_semantic_seg is a tuple. Unpacking...")
            #     gt_clip, loss_clip, acc_clip = gt_semantic_seg
            #     gt_semantic_seg = gt_clip
                
            # self.label_sanity_check(gt_semantic_seg)
            
            # # If there are unlabeled pixels
            # if not torch.all(gt_semantic_seg != 255):
            #     # Self-training path
            #     if self.self_train and self._iter_counter >= self.start_self_train[0] and \
            #         (self._iter_counter <= self.start_self_train[1] or self.start_self_train[1] < 0):
            #         #print(f"[DEBUG] Self-Training active at iter {self._iter_counter.item()}")
            #         with torch.no_grad():
            #             gt = gt_semantic_seg.clone()
            #             gt_self = self.assign_label(gt, feat,
            #                         self.norm_feat, self.unlabeled_cats)
            #             # 🔽 valid가 0이면 CLIP fallback 사용
            #             if (gt_self != -1).sum() == 0:
            #                 print("[DEBUG] gt_self empty, fallback to gt_clip")
            #                 gt_self = gt_clip
            #             del gt
            #     # CLIP-guided path
            #     if self.clip_guided and self._iter_counter >= self.start_clip_guided[0] and \
            #         (self._iter_counter <= self.start_clip_guided[1] or self.start_clip_guided[1] < 0):
            #         #print(f"[DEBUG] CLIP-Guided active at iter {self._iter_counter.item()}")
            #         with torch.no_grad():
            #             # clip cannot deal with background
            #             gt = gt_semantic_seg.clone()
            #             # Extract features from CLIP
            #             if gt_self is not None and self.cls_bg:
            #                 gt[gt_self == 0] = 0
            #             x = self.clip(img)[-1]
            #             q, k, v, cls_token = None, None, None, None
            #             if self.vit:
            #                 if isinstance(x, list) and len(x) == 4:
            #                     x, q, k, v = x
            #                 if isinstance(x, list) and len(x) == 2:
            #                     x, cls_token = x
            #                 if v is not None:
            #                     feat = self.proj(v)
            #                 else:
            #                     feat = self.proj(x)
            #                 if cls_token is not None:
            #                     cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
            #             else:
            #                 q = self.q_proj(x)
            #                 k = self.k_proj(x)
            #                 q = torch.flatten(q, start_dim=2).transpose(-2, -1)
            #                 k = torch.flatten(k, start_dim=2).transpose(-2, -1)
            #                 v = self.v_proj(x)
            #                 feat = self.c_proj(v)
            #             #print(f"[DEBUG] assigning clip-guided labels")
            #             gt_clip = self.assign_label(gt, feat,
            #                         True, self.clip_unlabeled_cats, 
            #                         k=k, cls_token=cls_token, clip=True)
            #             del gt
                
            #     # Final pseudo-label decision: self, clip, or hybrid        
            #     if gt_self is not None:
            #         #print("[DEBUG] unique labels after assign_label:", torch.unique(gt_clip)) 
            #         gt_semantic_seg = gt_self
            #     if gt_clip is not None:
            #         # merge gt_self and gt_clip
            #         if gt_self is not None:
            #             print("[DEBUG] NO gt_self !!")
            #             for i in self.trust_clip_on:
            #                 idx = (gt_clip == i)
            #                 gt_semantic_seg[idx] = i
            #         else:
            #             gt_semantic_seg = gt_clip
            #     # if gt_self is None or (gt_self == -1).all():
            #     #     print("[HYBRID] Fallback to CLIP-guided labels")
            #     #     gt_semantic_seg = gt_clip
            #     # else:
            #     #     gt_semantic_seg = gt_self
            #     #print("[DEBUG] unique labels after merging:", torch.unique(gt_semantic_seg))

            #     # ignore the unlabeled
            #     if isinstance(gt_semantic_seg, tuple):
            #         print("[DEBUG] gt_semantic_seg is a tuple. Unpacking before label assignment...")
                    
            #         if len(gt_semantic_seg) == 3:
            #             gt_clip, loss_clip, acc_clip = gt_semantic_seg
            #         elif len(gt_semantic_seg) == 2:
            #             gt_clip, acc_clip = gt_semantic_seg
            #             loss_clip = None  # or some dummy
            #         elif len(gt_semantic_seg) == 1:
            #             gt_clip = gt_semantic_seg[0]
            #             loss_clip, acc_clip = None, None
            #         else:
            #             raise ValueError(f"Unexpected gt_semantic_seg tuple length: {len(gt_semantic_seg)}")

            #         gt_semantic_seg = gt_clip

            #     gt_semantic_seg[gt_semantic_seg < 0] = 255

            gt_semantic_seg[gt_semantic_seg < 0] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            seg_logits = self.forward(inputs)
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]