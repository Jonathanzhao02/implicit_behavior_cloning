import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import MujocoDataset, MujocoTrajectoryDataset, mujoco_collate_fn
from models.basic_model import Backbone
from models.loss import InfoNCELoss
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# negative samples:
# progress: [0-1]
# pos: [-0.1-0.1]
# rot: [-1-1]

NAME = 'train-ibc-basic'
CKPT_PATH = 'ckpts/'
CKPT = 'ckpts/train-ibc-basic/100000.pth'
BATCH_SIZE = 32
N_SAMP = 8

EPOCHS = 10
EVAL_EVERY = 1
SAVE_EVERY = 10000

GLOBAL_STEP = 0

def generate(model, criterion, dataloader, device):
    global GLOBAL_STEP
    model.eval()

    for (img, sentence, ee_pos, ee_rot, joint_angles, gripper, progress) in dataloader:
        bsz = progress.size(0)
        img = img.to(device).repeat(N_SAMP, 1, 1, 1)
        sentence = sentence.to(device).repeat(N_SAMP, 1)
        
        ee_pos = ee_pos.to(device)
        ee_rot = ee_rot.to(device)
        joint_angles = joint_angles.to(device)
        gripper = gripper.to(device)
        progress = progress.to(device).unsqueeze(-1)
        action = torch.cat((ee_pos, ee_rot, gripper, progress), -1).float()

        c_ee_pos = (torch.rand(size=(bsz * N_SAMP,) + ee_pos.size()[1:], device=device) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=(bsz * N_SAMP,) + ee_rot.size()[1:], device=device) - 0.5) * 2
        c_gripper = (torch.rand(size=(bsz * N_SAMP,) + gripper.size()[1:], device=device) - 0.5) * 2
        c_progress = torch.rand(size=(bsz * N_SAMP,) + progress.size()[1:], device=device)
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper, c_progress), -1).float()
        c_action.requires_grad = True

        y = torch.ones(bsz * N_SAMP).to(device).unsqueeze(-1)

        optimizer = optim.Adam([c_action], lr=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

        for i in tqdm(range(EPOCHS)):
            optimizer.zero_grad()
            y_hat = model(img, sentence, c_action)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        import code
        code.interact(local=locals())

        GLOBAL_STEP += 1

if __name__ == '__main__':
    ckpt_path = Path(CKPT_PATH).joinpath(NAME)

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    model = Backbone()
    model = model.to(device)

    img_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dset = MujocoDataset(
        '/home/localuser/Documents/jjzhao/mujoco_ur5_robotiq85/data/train',
        img_preprocess=img_preprocess,
    )

    train_dataloader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=mujoco_collate_fn)

    val_dset = MujocoDataset(
        '/home/localuser/Documents/jjzhao/mujoco_ur5_robotiq85/data/val',
        img_preprocess=img_preprocess,
    )

    val_dataloader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=mujoco_collate_fn)

    # criterion = InfoNCELoss(N_NEG)
    criterion = nn.BCEWithLogitsLoss()

    if CKPT is not None:
        model.load_state_dict(torch.load(CKPT)['model'], strict=True)

    generate(model, criterion, train_dataloader, device)
        
