from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import clip
import h5py
import torch
import bisect
import torchvision
import numpy as np
import torch.nn.functional as F

def gen_sentence(attrs, targets):
    obj1 = targets.attrs['obj1']
    obj1 = targets.attrs['obj2']

    color1 = attrs['color'][obj1]
    color2 = attrs['color'][obj2]

    scale1 = attrs['scale'][obj1]
    scale2 = attrs['scale'][obj2]

    action = targets.attrs['action']

    raise NotImplementedError("Re-implement gen_sentence over new format!")

# Returns singular frames of demos
class MujocoDataset(Dataset):
    def __init__(self, data_dir):
        # |--data_dir
        #     |--demo0
        #         |--imgs
        #             |--0.png
        #             |--1.png
        #             |--...
        #         |--states.data
        #     |--demo1
        #     |--...

        self.data_dir = Path(data_dir)
        demos = [x for x in self.data_dir.iterdir() if x.is_dir()]
        self.demos = []
        self.demo_lens = []

        for i,demo in enumerate(demos):
            with h5py.File(demo.joinpath('states.data'), 'r') as f:
                if f.attrs['success']:
                    self.demos.append(demo)
                    self.demo_lens.append(f.attrs['final_timestep'] + (self.demo_lens[-1] if len(self.demo_lens) > 0 else 0))

    def __len__(self):
        return self.demo_lens[-1]

    def __getitem__(self, item):
        demo_idx = bisect.bisect_right(self.demo_lens, item)

        with h5py.File(self.demos[demo_idx].joinpath('states.data'), 'r') as f:
            if demo_idx > 0:
                step_idx = item - self.demo_lens[demo_idx - 1]
            else:
                step_idx = item
            
            demo_length = f.attrs['final_timestep']
            
            img = torchvision.io.read_image(str(self.demos[demo_idx].joinpath(f'imgs/{step_idx}.png'))).float() / 255

            # only support for one objective currently
            # sentence = gen_sentence(f['gen_attrs'], f['objectives']['0']['targets'])
            sentence = clip.tokenize("placeholder")[0]

            ee_pos = torch.tensor(f['pos'][step_idx][0])
            ee_rot = torch.tensor(f['rot'][step_idx][0])
            joint_angles = torch.tensor(f['q'][step_idx])
            progress = torch.tensor(step_idx / demo_length)

            return img, sentence, ee_pos, ee_rot, joint_angles, progress
        
        raise Exception(f'Failed to open file {str(demo.joinpath("states.data"))}')

# Returns trajectories generated from demos
class MujocoTrajectoryDataset(Dataset):
    def __init__(self, data_dir, sample_every=6, num_points=10):
        # |--data_dir
        #     |--demo0
        #         |--imgs
        #             |--0.png
        #             |--1.png
        #             |--...
        #         |--states.data
        #     |--demo1
        #     |--...

        self.data_dir = Path(data_dir)
        demos = [x for x in self.data_dir.iterdir() if x.is_dir()]
        self.demos = []
        self.demo_lens = []
        self.sample_every = sample_every
        self.num_points = num_points

        for i,demo in enumerate(demos):
            with h5py.File(demo.joinpath('states.data'), 'r') as f:
                if f.attrs['success']:
                    self.demos.append(demo)
                    self.demo_lens.append(f.attrs['final_timestep'] + (self.demo_lens[-1] if len(self.demo_lens) > 0 else 0))

    def __len__(self):
        return self.demo_lens[-1]

    def __getitem__(self, item):
        demo_idx = bisect.bisect_right(self.demo_lens, item)

        with h5py.File(self.demos[demo_idx].joinpath('states.data'), 'r') as f:
            if demo_idx > 0:
                step_idx = item - self.demo_lens[demo_idx - 1]
            else:
                step_idx = item
            
            demo_length = f.attrs['final_timestep']
            
            img = torchvision.io.read_image(str(self.demos[demo_idx].joinpath(f'imgs/{step_idx}.png'))).float() / 255

            # only support for one objective currently
            # sentence = gen_sentence(f['gen_attrs'], f['objectives']['0']['targets'])
            sentence = clip.tokenize("placeholder")[0]

            ee_pos = torch.tensor(f['pos'][step_idx:demo_length:self.sample_every][:self.num_points][:,0])
            ee_rot = torch.tensor(f['rot'][step_idx:demo_length:self.sample_every][:self.num_points][:,0])
            joint_angles = torch.tensor(f['q'][step_idx:demo_length:self.sample_every][:self.num_points])

            diff = self.num_points - ee_pos.shape[0]

            if diff > 0:
                f_ee_pos = ee_pos[-1]
                f_ee_rot = ee_rot[-1]
                f_joint_angles = joint_angles[-1]
                ee_pos = F.pad(ee_pos, (0, 0, 0, diff))
                ee_rot = F.pad(ee_rot, (0, 0, 0, diff))
                joint_angles = F.pad(joint_angles, (0, 0, 0, diff))
                ee_pos[self.num_points - diff:] = f_ee_pos
                ee_rot[self.num_points - diff:] = f_ee_rot
                joint_angles[self.num_points - diff:] = f_joint_angles

            progress = torch.clamp((step_idx + torch.arange(0, self.num_points) * self.sample_every) / demo_length, max=1)

            return img, sentence, ee_pos, ee_rot, joint_angles, progress
        
        raise Exception(f'Failed to open file {str(demo.joinpath("states.data"))}')

def mujoco_collate_fn(batch):
    (img, sentence, ee_pos, ee_rot, joint_angles, progress) = zip(*batch)

    img = torch.stack(img)
    sentence = torch.stack(sentence)
    ee_pos = torch.stack(ee_pos)
    ee_rot = torch.stack(ee_rot)
    joint_angles = torch.stack(joint_angles)
    progress = torch.stack(progress)

    return img, sentence, ee_pos, ee_rot, joint_angles, progress
