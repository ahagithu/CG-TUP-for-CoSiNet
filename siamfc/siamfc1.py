from __future__ import absolute_import, division, print_function
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import torchvision.models.vgg

from torch.utils import model_zoo
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

import sys

sys.path.append(os.path.abspath('.'))

from siamfc import ops
from siamfc.heads import SiamFC
from siamfc.losses import GHMCLoss
from siamfc.losses import FocalLoss
from siamfc.losses import BalancedLoss
from siamfc.datasets import Pair
from siamfc.transforms import SiamFCTransforms
from siamfc import backbones

from siamfc.attention import GlobalAttentionBlock, CBAM
from siamfc.backbones import SELayer1, ECALayer, ECALayer1
from siamfc.dcn import DeformConv2d
from siamfc.psp import PSA

__all__ = ['TrackerSiamFC']


# Main network structure
class Net(nn.Module):
    def __init__(self, backbone, backbone1, head):
        super(Net, self).__init__()
        self.head = head
        self.backbone1 = backbone1
        self.backbone = backbone
        self.conv = nn.Sequential(nn.Conv2d(2, 1, 1))

        # Attention modules
        self.tematt = ECALayer(256)  # Template attention
        self.detatt = CBAM(256)  # Detection attention
        self.attse = ECALayer1(256)  # Channel attention

    def forward(self, z, x):
        # Extract features from both backbones
        z1 = self.backbone(z)  # VGG features
        x1 = self.backbone(x)
        z2 = self.backbone1(z)  # AlexNet features
        x2 = self.backbone1(x)

        # Apply attention to template features
        zf1 = self.tematt(z1)
        zf11 = self.attse(z1)
        zf11 = zf11 + z1
        z1 = zf1 + zf11

        zf2 = self.tematt(z2)
        zf22 = self.attse(z2)
        zf22 = zf22 + z2
        z2 = zf2 + zf22

        # Apply attention to detection features
        xf1 = self.detatt(x1)
        xf11 = self.attse(x1)
        xf11 = xf11 + x1
        x1 = xf1 + xf11

        xf2 = self.detatt(x2)
        xf22 = self.attse(x2)
        xf22 = xf22 + x2
        x2 = xf2 + xf22

        # Get responses from both backbones
        out1 = self.head(z1, x1)
        out2 = self.head(z2, x2)

        # Fusion strategy
        out11 = out1 + out2  # Additive fusion
        out22 = out1 * out2  # Multiplicative fusion
        out = torch.cat([out11, out22], dim=1)
        out = self.conv(out)  # Final fusion

        return out


class TrackerSiamFC(Tracker):
    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # Setup model
        self.net = Net(
            backbone=backbones.vgg(),
            backbone1=backbones.AlexNetV(),
            head=SiamFC(self.cfg.out_scale),
        )
        ops.init_weights(self.net)

        # Load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path,
                map_location=lambda storage, loc: storage),
                strict=False)
        self.net = self.net.to(self.device)

        # Template update variables
        self.frame_count = 0
        self.best_response = 0
        self.best_template = None
        self.current_template_z1 = None
        self.current_template_z2 = None
        self.response_history = []
        self.update_log = []  # Store update history

        # Setup criterion
        self.criterion = BalancedLoss()

        # Setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # Setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # Default parameters
        cfg = {
            # Basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,

            # Inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,

            # Training parameters
            'epoch_num': 50,
            'batch_size': 16,
            'num_workers': 8,
            'initial_lr': 1e-3,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,

            # Template update parameters
            'template_update_interval': 25,
            'template_learning_rate': 0.026,
            'quality_threshold': 0.012,
            'response_normalization_factor': 12.0,
            'show_update_conditions': False,

            # Advanced template update parameters
            'adaptive_update': True,
            'max_updates_per_sequence': 20,
            'min_quality_gain': 0.05,
            'emergency_update_threshold': 0.009,

            # Ablation study parameters for template update
            'disable_interval_condition': False,  # Disable interval condition
            'disable_confidence_condition': False,  # Disable confidence condition
            'disable_adaptive_condition': False,  # Disable adaptive condition
            'force_update_every_frame': False,  # Force update every frame
            'handle_occlusions': True, # set to true if you want to enable it
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def _update_template(self, img, max_response, scale_id):
        """Template update based on tracking quality"""
        self.frame_count += 1

        # Store response history
        self.response_history.append(max_response)
        if len(self.response_history) > 20:
            self.response_history.pop(0)

        # Calculate metrics for decision
        normalized_response = max_response / self.cfg.response_normalization_factor
        avg_response = np.mean(self.response_history) if self.response_history else max_response
        adaptive_threshold = avg_response * 0.7

        # Conditions for update
        if self.cfg.force_update_every_frame:
            # Force update every frame (ablation study)
            update_condition = True
            condition_interval = True
            condition_confidence = True
            condition_adaptive = True
        else:
            # Normal conditions with ablation options
            condition_interval = (self.frame_count % self.cfg.template_update_interval == 0)
            condition_confidence = (max_response > self.cfg.quality_threshold)
            condition_adaptive = (max_response > adaptive_threshold)

            # Apply ablation settings
            if self.cfg.disable_interval_condition:
                condition_interval = True  # Always true = condition disabled
            if self.cfg.disable_confidence_condition:
                condition_confidence = True  # Always true = condition disabled
            if self.cfg.disable_adaptive_condition:
                condition_adaptive = True  # Always true = condition disabled

            update_condition = (
                    condition_interval and
                    condition_confidence and
                    condition_adaptive
            )

        # Show conditions if requested
        if self.cfg.show_update_conditions:
            ablation_info = []
            if self.cfg.disable_interval_condition:
                ablation_info.append("interval_disabled")
            if self.cfg.disable_confidence_condition:
                ablation_info.append("confidence_disabled")
            if self.cfg.disable_adaptive_condition:
                ablation_info.append("adaptive_disabled")
            if self.cfg.force_update_every_frame:
                ablation_info.append("force_update")

            print(f"Frame {self.frame_count}: Response={max_response:.3f}, "
                  f"Normalized={normalized_response:.3f}, "
                  f"Avg={avg_response:.3f}, "
                  f"AdaptThresh={adaptive_threshold:.3f}, "
                  f"Interval={condition_interval}, "
                  f"Confidence={condition_confidence}, "
                  f"Adaptive={condition_adaptive}")
            if ablation_info:
                print(f"Ablation settings: {', '.join(ablation_info)}")

        if update_condition:
            try:
                # Extract new template
                z_new = ops.crop_and_resize(
                    img, self.center, self.z_sz,
                    out_size=self.cfg.exemplar_sz,
                    border_value=self.avg_color)

                z_new = torch.from_numpy(z_new).to(
                    self.device).permute(2, 0, 1).unsqueeze(0).float()

                # Calculate new features
                z1_new = self.net.backbone(z_new)
                z2_new = self.net.backbone1(z_new)

                # Apply attention to new template
                zf1_new = self.net.tematt(z1_new)
                zf11_new = self.net.attse(z1_new)
                zf11_new = zf11_new + z1_new
                kernel1_new = zf1_new + zf11_new

                zf2_new = self.net.tematt(z2_new)
                zf22_new = self.net.attse(z2_new)
                zf22_new = zf22_new + z2_new
                kernel2_new = zf2_new + zf22_new

                # Linear update
                lr = self.cfg.template_learning_rate
                self.kernel1 = (1 - lr) * self.kernel1 + lr * kernel1_new
                self.kernel2 = (1 - lr) * self.kernel2 + lr * kernel2_new

                # Log update
                update_info = {
                    'frame': self.frame_count,
                    'response': max_response,
                    'normalized_response': normalized_response,
                    'adaptive_threshold': adaptive_threshold,
                    'learning_rate': lr,
                    'ablation_type': 'normal'
                }
                if self.cfg.force_update_every_frame:
                    update_info['ablation_type'] = 'force_update'
                elif self.cfg.disable_interval_condition:
                    update_info['ablation_type'] = 'no_interval'
                elif self.cfg.disable_confidence_condition:
                    update_info['ablation_type'] = 'no_confidence'
                elif self.cfg.disable_adaptive_condition:
                    update_info['ablation_type'] = 'no_adaptive'

                self.update_log.append(update_info)

                print(f"✅ Template updated at frame {self.frame_count}, "
                      f"response: {max_response:.3f}, "
                      f"normalized: {normalized_response:.3f}, "
                      f"adaptive_threshold: {adaptive_threshold:.3f}")

            except Exception as e:
                print(f"❌ Template update failed: {e}")
        else:
            if self.cfg.show_update_conditions and not self.cfg.force_update_every_frame:
                reasons = []
                if not condition_interval:
                    reasons.append(f"interval (need frame % {self.cfg.template_update_interval} == 0)")
                if not condition_confidence:
                    reasons.append(f"confidence (need > {self.cfg.quality_threshold:.3f})")
                if not condition_adaptive:
                    reasons.append(f"adaptive (need > {adaptive_threshold:.3f})")

                if reasons:
                    print(f"⏩ Skip update at frame {self.frame_count}: " + ", ".join(reasons))

    def _handle_occlusions(self, max_response):
        """Handle occlusions to avoid incorrect updates"""
        if len(self.response_history) > 5:
            avg_response = np.mean(self.response_history)
            occlusion_threshold = avg_response * 0.4 # 3.0 default value
        else:
            occlusion_threshold = 5.0 # 5.0 default value

        occlusion_detected = max_response < occlusion_threshold

        if occlusion_detected and self.cfg.show_update_conditions:
            print(f"🚫 Occlusion detected at frame {self.frame_count}, "
                  f"response: {max_response:.3f}, "
                  f"threshold: {occlusion_threshold:.3f}")

        return occlusion_detected

    def get_update_statistics(self):
        """Get template update statistics"""
        if not self.update_log:
            return "No template updates performed"

        responses = [log['response'] for log in self.update_log]
        frames = [log['frame'] for log in self.update_log]
        ablation_types = [log.get('ablation_type', 'normal') for log in self.update_log]

        # Count by ablation type
        from collections import Counter
        type_counter = Counter(ablation_types)

        stats = (f"Template Update Statistics:\n"
                 f"  Total updates: {len(self.update_log)}\n"
                 f"  Update frames: {frames}\n"
                 f"  By ablation type: {dict(type_counter)}\n"
                 f"  Average response: {np.mean(responses):.3f}\n"
                 f"  Minimum response: {np.min(responses):.3f}\n"
                 f"  Maximum response: {np.max(responses):.3f}")

        return stats

    @torch.no_grad()
    def init(self, img, box):
        """Initialize tracker with first frame"""
        self.net.eval()

        # Convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # Create Hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # Search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2,
            self.cfg.scale_num)

        # Exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # Exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # Exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        z1 = self.net.backbone(z)
        z2 = self.net.backbone1(z)

        # Apply attention to initial template
        zf1 = self.net.tematt(z1)
        zf11 = self.net.attse(z1)
        zf11 = zf11 + z1
        self.kernel1 = zf1 + zf11

        zf2 = self.net.tematt(z2)
        zf22 = self.net.attse(z2)
        zf22 = zf22 + z2
        self.kernel2 = zf2 + zf22

        # Store initial template
        self.current_template_z1 = self.kernel1.clone()
        self.current_template_z2 = self.kernel2.clone()
        self.best_response = 1.0
        self.frame_count = 0
        self.response_history = [5.0]  # Initial value
        self.update_log = []

        if self.cfg.show_update_conditions:
            print(f"🎯 Initialization at frame 0 - Template ready for tracking")
            if self.cfg.force_update_every_frame:
                print(f"⚙️  Ablation mode: Force update every frame")
            elif self.cfg.disable_interval_condition:
                print(f"⚙️  Ablation mode: Interval condition disabled")
            elif self.cfg.disable_confidence_condition:
                print(f"⚙️  Ablation mode: Confidence condition disabled")
            elif self.cfg.disable_adaptive_condition:
                print(f"⚙️  Ablation mode: Adaptive condition disabled")

    @torch.no_grad()
    def update(self, img):
        """Update tracker with new frame"""
        self.net.eval()

        # Search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # Extract features
        x1 = self.net.backbone(x)
        x2 = self.net.backbone1(x)

        # Apply attention to detection features
        xf1 = self.net.detatt(x1)
        xf11 = self.net.attse(x1)
        xf11 = xf11 + x1
        x1 = xf1 + xf11

        xf2 = self.net.detatt(x2)
        xf22 = self.net.attse(x2)
        xf22 = xf22 + x2
        x2 = xf2 + xf22

        # Get responses
        responses1 = self.net.head(self.kernel1, x1)
        responses2 = self.net.head(self.kernel2, x2)

        # Fusion strategy
        out1 = responses1
        out2 = responses2
        responses1 = out1 + out2
        responses2 = out1 * out2
        responses = self.net.conv(torch.cat([responses1, responses2], dim=1))

        responses = responses.squeeze(1).cpu().numpy()

        # Upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])

        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # Peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # Peak location
        response = responses[scale_id]
        max_response = response.max()

        # Handle occlusions and update template
        if self.cfg.handle_occlusions:
            occluded = self._handle_occlusions(max_response)
        else:
            occluded = False

        if not occluded:
            self._update_template(img, max_response, scale_id)

        # Process response
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window

        # Find peak location
        loc = np.unravel_index(response.argmax(), response.shape)

        # Locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz

        self.center += disp_in_image

        # Update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # Return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def track(self, img_files, box, visualize=False):
        """Track object through image sequence"""
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        # Display final statistics
        if self.cfg.show_update_conditions:
            print("\n" + "=" * 50)
            print(self.get_update_statistics())
            print("=" * 50)

        return boxes, times

    def train_step(self, batch, backward=True):
        """Single training step"""
        self.net.train(backward)

        # Parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # Inference
            responses = self.net(z, x)

            # Calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # Back propagation
                self.optimizer.zero_grad()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                loss.backward()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None, save_dir='models'):
        """Complete training procedure"""
        self.net.train()

        # Create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = Pair(seqs=seqs, transforms=transforms)

        # Setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # Loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # Loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # Save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

            self.lr_scheduler.step(epoch=epoch)

    def _create_labels(self, size):
        """Create labels for training"""
        # Skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # Distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # Create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # Repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # Convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels


# Utility functions for ablation experiments
def create_ablation_configurations():
    """Create different ablation study configurations"""
    configs = {
        # Baseline: all conditions enabled
        'baseline': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 1: No interval condition
        'no_interval': {
            'disable_interval_condition': True,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 2: No confidence condition
        'no_confidence': {
            'disable_interval_condition': False,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 3: No adaptive condition
        'no_adaptive': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 4: Force update every frame
        'force_update': {
            'disable_interval_condition': False,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': False,
            'force_update_every_frame': True,
            'show_update_conditions': True,
        },

        # Ablation 5: Only interval condition
        'only_interval': {
            'disable_interval_condition': False,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 6: Only confidence condition
        'only_confidence': {
            'disable_interval_condition': True,
            'disable_confidence_condition': False,
            'disable_adaptive_condition': True,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },

        # Ablation 7: Only adaptive condition
        'only_adaptive': {
            'disable_interval_condition': True,
            'disable_confidence_condition': True,
            'disable_adaptive_condition': False,
            'force_update_every_frame': False,
            'show_update_conditions': True,
        },
    }

    return configs


def run_ablation_experiment(sequence_path, initial_bbox, model_path, config_name, config):
    """Run a single ablation experiment"""
    print(f"\n{'=' * 60}")
    print(f"Running ablation experiment: {config_name}")
    print(f"Configuration: {config}")
    print(f"{'=' * 60}")

    # Create tracker with specific configuration
    tracker = TrackerSiamFC(
        net_path=model_path,
        **config  # Pass all configuration parameters
    )

    # Run tracking
    boxes, times = tracker.track(sequence_path, initial_bbox)

    # Get statistics
    stats = tracker.get_update_statistics()

    return {
        'config_name': config_name,
        'config': config,
        'boxes': boxes,
        'times': times,
        'statistics': stats,
        'update_log': tracker.update_log
    }


def compare_ablation_results(results):
    """Compare results from different ablation experiments"""
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY RESULTS COMPARISON")
    print(f"{'=' * 80}")

    for result in results:
        print(f"\nExperiment: {result['config_name']}")
        print(f"Configuration: {result['config']}")
        print(f"Total updates: {len(result['update_log'])}")

        if result['update_log']:
            responses = [log['response'] for log in result['update_log']]
            print(f"Average response: {np.mean(responses):.3f}")
            print(f"Update frames: {[log['frame'] for log in result['update_log']]}")

        print("-" * 40)