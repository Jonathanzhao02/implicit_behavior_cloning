from torch.utils.data import DataLoader
from utils.data import MujocoDataset, MujocoTrajectoryDataset, mujoco_collate_fn

if __name__ == '__main__':
    dset = MujocoTrajectoryDataset(
        '/home/jjzhao/Documents/IRLab/mujoco_ur5_robotiq85/data',
    )

    dataloader = DataLoader(dset, batch_size=10, shuffle=True, num_workers=1, collate_fn=mujoco_collate_fn)

    for i,data in enumerate(dataloader):
        print(f'batch {i}')
        import code
        code.interact(local=locals())
