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
# gripper: [-1-1]
# rot: [-1-1]

NAME = 'train-ibc-basic'
CKPT_PATH = 'ckpts/'
CKPT = '/home/localuser/Documents/jjzhao/implicit_behavior_cloning/ckpts/train-ibc-basic0/290000.pth'
BATCH_SIZE = 32
N_NEG = 8

EPOCHS = 300
EVAL_EVERY = 1
SAVE_EVERY = 10000

GLOBAL_STEP = 0

def train(model, optimizer, scheduler, criterion, dataloader, writer, device, ckpt_path):
    global GLOBAL_STEP
    model.train()

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
        c_gripper = (torch.rand(size=(bsz * N_NEG,) + gripper.size()[1:], device=device) - 0.5) * 2
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper), -1).float()

        action = torch.cat((action, c_action), 0)
        y = torch.cat((torch.ones(bsz), torch.zeros(bsz * N_NEG))).to(device).unsqueeze(-1)

        optimizer.zero_grad()
        y_hat = model(img, sentence, action)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('train-loss', loss.item(), global_step=GLOBAL_STEP)

        if GLOBAL_STEP % SAVE_EVERY == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, ckpt_path.joinpath(f'{GLOBAL_STEP}.pth'))

        GLOBAL_STEP += 1

def test(model, criterion, dataloader, writer, device):
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
            c_gripper = (torch.rand(size=(bsz * N_NEG,) + gripper.size()[1:], device=device) - 0.5) * 2
            c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper), -1).float()

            action = torch.cat((action, c_action), 0)
            y = torch.cat((torch.ones(bsz), torch.zeros(bsz * N_NEG))).to(device).unsqueeze(-1)

            y_hat = model(img, sentence, action)
            loss = criterion(y_hat, y)
            total_loss += loss.item()

        writer.add_scalar('test-loss', total_loss, global_step=GLOBAL_STEP)

if __name__ == '__main__':
    writer = SummaryWriter('runs/' + NAME)
    ckpt_path = Path(CKPT_PATH).joinpath(NAME)
    ckpt_path.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Backbone(action_size=10)
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

    criterion = InfoNCELoss(N_NEG)
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

    if CKPT is not None:
        model.load_state_dict(torch.load(CKPT)['model'], strict=True)
        optimizer.load_state_dict(torch.load(CKPT)['optimizer'])
        test(model, criterion, val_dataloader, writer, device)

    for i in range(EPOCHS):
        print(f"Training epoch {i}")
        train(model, optimizer, scheduler, criterion, train_dataloader, writer, device, ckpt_path)
        
        if i % EVAL_EVERY == 0:
            test(model, criterion, val_dataloader, writer, device)

