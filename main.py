import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import MujocoDataset, MujocoTrajectoryDataset, mujoco_collate_fn
from models.basic_model import Backbone
from models.loss import InfoNCELoss
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# negative samples:
# progress: [0-1]
# pos: [-0.1-0.1]
# rot: [-1-1]

NAME = 'train-ibc-basic'
BATCH_SIZE = 32
N_NEG = 8

GLOBAL_STEP = 0

def train(model, epoch, optimizer, scheduler, criterion, dataloader, writer, device):
    global GLOBAL_STEP
    model.train()

    for (img, sentence, ee_pos, ee_rot, joint_angles, progress) in tqdm(dataloader):
        img = img.to(device).repeat(1 + N_NEG, 1, 1, 1)
        sentence = sentence.to(device).repeat(1 + N_NEG, 1)
        
        ee_pos = ee_pos.to(device)
        ee_rot = ee_rot.to(device)
        joint_angles = joint_angles.to(device)
        progress = progress.to(device).unsqueeze(-1)
        action = torch.cat((ee_pos, ee_rot, progress), -1).float()

        c_ee_pos = (torch.rand(size=(BATCH_SIZE * N_NEG,) + ee_pos.size()[1:], device=device) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=(BATCH_SIZE * N_NEG,) + ee_rot.size()[1:], device=device) - 0.5) * 2
        c_progress = torch.rand(size=(BATCH_SIZE * N_NEG,) + progress.size()[1:], device=device)
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_progress), -1).float()

        action = torch.cat((action, c_action), 0)
        y = torch.cat((torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE * N_NEG))).to(device)

        optimizer.zero_grad()
        y_hat = model(img, sentence, action)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('train-loss', loss.item(), global_step=GLOBAL_STEP)

def test(model, criterion, dataloader, writer, device):
    with torch.no_grad():
        model.eval()
        total_loss = 0

        for (img, sentence, ee_pos, ee_rot, joint_angles, progress) in tqdm(dataloader):
            img = img.to(device).repeat(1 + N_NEG, 1, 1, 1)
            sentence = sentence.to(device).repeat(1 + N_NEG, 1)
            
            ee_pos = ee_pos.to(device)
            ee_rot = ee_rot.to(device)
            joint_angles = joint_angles.to(device)
            progress = progress.to(device).unsqueeze(-1)
            action = torch.cat((ee_pos, ee_rot, progress), -1).float()

            c_ee_pos = (torch.rand(size=(BATCH_SIZE * N_NEG,) + ee_pos.size()[1:], device=device) - 0.5) * 0.2
            c_ee_rot = (torch.rand(size=(BATCH_SIZE * N_NEG,) + ee_rot.size()[1:], device=device) - 0.5) * 2
            c_progress = torch.rand(size=(BATCH_SIZE * N_NEG,) + progress.size()[1:], device=device)
            c_action = torch.cat((c_ee_pos, c_ee_rot, c_progress), -1).float()

            action = torch.cat((action, c_action), 0)
            y = torch.cat((torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE * N_NEG))).to(device)

            y_hat = model(img, sentence, action)
            loss = criterion(y_hat, y)
            total_loss += loss.item()

        writer.add_scalar('test-loss', total_loss, global_step=GLOBAL_STEP)

if __name__ == '__main__':
    writer = SummaryWriter('runs/' + NAME)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Backbone(device=device).to(device)

    img_preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dset = MujocoDataset(
        '/home/jjzhao/Documents/IRLab/mujoco_ur5_robotiq85/data',
        img_preprocess=img_preprocess,
    )

    dataloader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=mujoco_collate_fn)

    criterion = InfoNCELoss(BATCH_SIZE, N_NEG)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
