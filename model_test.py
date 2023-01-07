import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import MujocoDataset, MujocoTrajectoryDataset, mujoco_collate_fn
from models.film_model import Backbone
from models.loss import InfoNCELoss
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

NAME = 'train-ibc-film-test-tmp'
CKPT_PATH = 'ckpts/'
CKPT = '/home/localuser/Documents/jjzhao/implicit_behavior_cloning/ckpts/train-ibc-film/500.pth'
BATCH_SIZE = 2
N_NEG = 3

GLOBAL_STEP = 0

def test(model, criterion, dataloader, writer, device):
    global GLOBAL_STEP
    with torch.no_grad():
        model.eval()

        total_loss = 0

        for (img, sentence, ee_pos, ee_rot, joint_angles, gripper, progress) in tqdm(dataloader):
            bsz = progress.size(0)
            img = img.to(device).repeat(1 + N_NEG, 1, 1, 1)
            sentence = sentence.to(device).repeat(1 + N_NEG, 1)
            
            ee_pos = ee_pos.to(device)
            ee_rot = ee_rot.to(device)
            joint_angles = joint_angles.to(device)
            gripper = gripper.unsqueeze(-1).to(device)
            action = torch.cat((ee_pos, ee_rot, gripper), -1).float()

            c_ee_pos = (torch.rand(size=(bsz * N_NEG,) + ee_pos.size()[1:], device=device) - 0.5) * 0.2
            c_ee_rot = (torch.rand(size=(bsz * N_NEG,) + ee_rot.size()[1:], device=device) - 0.5) * 2
            # c_gripper = (torch.rand(size=(bsz * N_NEG,) + gripper.size()[1:], device=device) - 0.5) * 2
            c_gripper = torch.tensor(np.random.choice([-0.12, 0.12], size=(bsz * N_NEG,) + gripper.size()[1:]), device=device)
            c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper), -1).float()

            action = torch.cat((action, c_action), 0)
            y = torch.cat((torch.ones(bsz), torch.zeros(bsz * N_NEG))).to(device).unsqueeze(-1)

            y_hat = model(img, sentence, action)
            loss = criterion(y_hat, y)

            y_hat = y_hat.squeeze()

            writer.add_scalars('test-output', { f'y-hat-{k}': y_hat[k].item() for k in range(y_hat.numel()) }, global_step=GLOBAL_STEP)
            writer.add_scalar('test-loss', loss.item(), global_step=GLOBAL_STEP)
            GLOBAL_STEP += 1

if __name__ == '__main__':
    writer = SummaryWriter('runs/' + NAME)
    ckpt_path = Path(CKPT_PATH).joinpath(NAME)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Backbone(action_size=100)
    model = model.to(device)

    img_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dset = MujocoTrajectoryDataset(
        '/home/localuser/Documents/jjzhao/mujoco_ur5_robotiq85/data/val',
        img_preprocess=img_preprocess,
    )

    val_dataloader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=mujoco_collate_fn)

    criterion = InfoNCELoss(N_NEG)
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    if CKPT is not None:
        model.load_state_dict(torch.load(CKPT)['model'], strict=True)
        optimizer.load_state_dict(torch.load(CKPT)['optimizer'])
        test(model, criterion, val_dataloader, writer, device)
