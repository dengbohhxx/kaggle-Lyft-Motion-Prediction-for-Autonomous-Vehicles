#!/usr/bin/env python
# coding: utf-8

# # Lyft: Prediction with multi-mode confidence
# 
# Thank you for the original training and inference kernel [@corochann](https://www.kaggle.com/corochann), don't forget to upvote those!  
# This kernel here is just a demonstration of the capabilities of these kernels below. With just very few changes in the training routine (e.g. increasing the rasterizer to 448 x 448 px) the model achieves a public LB score of 33.953, which is a 28% reduction to the original model (original LB score: 47.068).
# - https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence  
# - https://www.kaggle.com/corochann/lyft-prediction-with-multi-mode-confidence  

# # Environment setup
# 
#  - Please add [pestipeti/lyft-l5kit-unofficial-fix](https://www.kaggle.com/pestipeti/lyft-l5kit-unofficial-fix) as utility script.
#     - Official utility script "[philculliton/kaggle-l5kit](https://www.kaggle.com/mathurinache/kaggle-l5kit)" does not work with pytorch GPU.

# In[ ]:


import os
from pathlib import Path

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

# --- setup ---
pd.set_option('max_columns', 50)


# In[ ]:


import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv

print("l5kit version:", l5kit.__version__)


# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.models import resnet18
from torch import nn
from typing import Dict


# ## Model
# 
# pytorch model definition. Here model outputs both **multi-mode trajectory prediction & confidence of each trajectory**.

# In[ ]:


# --- Model utils ---
class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # TODO: support other than resnet18?
        backbone = resnet18(pretrained=False, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    


# In[ ]:


# --- Utils ---
import yaml


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    


# In[ ]:


# Referred https://www.kaggle.com/pestipeti/pytorch-baseline-inference
def run_prediction(predictor, data_loader):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(device)
            # target_availabilities = data["target_availabilities"].to(device)
            # targets = data["target_positions"].to(device)
            pred, confidences = predictor(image)

            pred_coords_list.append(pred.cpu().numpy().copy())
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs


# ## Configs

# In[ ]:


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [448, 448],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 12,
        'shuffle': True,
        'num_workers': 4
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
    },

    'train_params': {
        'max_num_steps': 10000,
        'checkpoint_every_n_steps': 5000,

        # 'eval_every_n_steps': -1
    }
}


# In[ ]:


flags_dict = {
    "debug": False,
    # --- Data configs ---
    "l5kit_data_folder": "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}


# # Main script
# 
# Now finished defining all the util codes. Let's start writing main script to train the model!

# ## Loading data
# 
# Here we will only use the first dataset from the sample set. (sample.zarr data is used for visualization, please use train.zarr / validate.zarr / test.zarr for actual model training/validation/prediction.)<br/>
# We're building a `LocalDataManager` object. This will resolve relative paths from the config using the `L5KIT_DATA_FOLDER` env variable we have just set.

# In[ ]:


flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")
save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)
debug = flags.debug


# In[ ]:


# set env variable for data
l5kit_data_folder = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
default_test_cfg = {
    'key': 'scenes/test.zarr',
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
}
test_cfg = cfg.get("test_data_loader", default_test_cfg)

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

test_path = test_cfg["key"]
print(f"Loading from {test_path}")
test_zarr = ChunkedDataset(dm.require(test_path)).open()
print("test_zarr", type(test_zarr))
test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]
test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataset = test_agent_dataset
if debug:
    # Only use 100 dataset for fast check...
    test_dataset = Subset(test_dataset, np.arange(100))
test_loader = DataLoader(
    test_dataset,
    shuffle=test_cfg["shuffle"],
    batch_size=test_cfg["batch_size"],
    num_workers=test_cfg["num_workers"],
    pin_memory=True,
)

print(test_agent_dataset)
print("# AgentDataset test:", len(test_agent_dataset))
print("# ActualDataset test:", len(test_dataset))


# ## Prepare model & optimizer

# In[ ]:


device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
else:
    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

pt_path = "/kaggle/input/lyft-prediction-public-models/multi_mode_448px.pth"
print(f"Loading from {pt_path}")

import copy
state_dict = torch.load(pt_path)
state_dict_v2 = copy.deepcopy(state_dict)

for key in state_dict:
    new_key = key[10:]
    state_dict_v2[new_key] = state_dict_v2.pop(key)

predictor.load_state_dict(state_dict_v2)
predictor.to(device)


# # Inference!

# In[ ]:


# --- Inference ---
timestamps, track_ids, coords, confs = run_prediction(predictor, test_loader)


# In[ ]:


csv_path = "submission.csv"
write_pred_csv(
    csv_path,
    timestamps=timestamps,
    track_ids=track_ids,
    coords=coords,
    confs=confs)
print(f"Saved to {csv_path}")

