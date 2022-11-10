import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import MujocoDataset, MujocoTrajectoryDataset, mujoco_collate_fn
from models.basic_model import Backbone

from tqdm.auto import tqdm

# negative samples:
# progress: [0-1]
# pos: [-0.1-0.1]
# rot: [-1-1]

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Backbone(device=device).to(device)

    img_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dset = MujocoDataset(
        '/home/jjzhao/Documents/IRLab/mujoco_ur5_robotiq85/data',
        img_preprocess=img_preprocess,
    )

    dataloader = DataLoader(dset, batch_size=256, shuffle=True, num_workers=1, collate_fn=mujoco_collate_fn)

    for (img, sentence, ee_pos, ee_rot, joint_angles, progress) in tqdm(dataloader):
        img = img.to(device).repeat(2, 1, 1, 1)
        sentence = sentence.to(device).repeat(2, 1)
        
        ee_pos = ee_pos.to(device)
        ee_rot = ee_rot.to(device)
        joint_angles = joint_angles.to(device)
        progress = progress.to(device).unsqueeze(-1)

        action = torch.cat((ee_pos, ee_rot, progress), -1).float()

        c_ee_pos = (torch.rand(size=ee_pos.size(), device=device) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=ee_rot.size(), device=device) - 0.5) * 2
        c_progress = torch.rand(size=progress.size(), device=device)

        c_action = torch.cat((c_ee_pos, c_ee_rot, c_progress), -1).float()

        action = torch.cat((action, c_action), 0)
