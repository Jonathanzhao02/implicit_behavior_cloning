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

    model = Backbone().to(device)

    img_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dset = MujocoDataset(
        '/home/jjzhao/Documents/IRLab/mujoco_ur5_robotiq85/data',
        img_preprocess=img_preprocess,
    )

    dataloader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=1, collate_fn=mujoco_collate_fn)

    for (img, sentence, ee_pos, ee_rot, joint_angles, progress) in tqdm(dataloader):
        img = img.to(device)
        sentence = sentence.to(device)
        
        ee_pos = ee_pos.to(device)
        ee_rot = ee_rot.to(device)
        joint_angles = joint_angles.to(device)
        progress = progress.to(device).unsqueeze(-1)

        action = torch.cat((ee_pos, ee_rot, progress), -1).float()
