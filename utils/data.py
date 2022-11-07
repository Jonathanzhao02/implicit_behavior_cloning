from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import clip
import h5py
import bisect
import torchvision

def gen_sentence(attrs, targets):
    obj1 = targets.attrs['obj1']
    obj1 = targets.attrs['obj2']

    color1 = attrs['color'][obj1]
    color2 = attrs['color'][obj2]

    scale1 = attrs['scale'][obj1]
    scale2 = attrs['scale'][obj2]

class MujocoDataset(Dataset):
    def __init__(self, data_dir):
        # |--data_dir
        #     |--demo0
        #         |--imgs
        #             |--img0.png
        #             |--img1.png
        #             |--...
        #         |--states.data
        #     |--demo1
        #     |--...

        self.data_dir = Path(data_dir)
        self.demos = [x for x in self.data_dir.iterdir() if x.is_dir()]
        self.demo_lens = np.zeros(len(self.demos))

        for i,demo in enumerate(self.demos):
            with h5py.File(demo.joinpath('states.data'), 'r') as f:
                self.demo_lens[i] = f.attrs['final_timestep'] + self.demo_lens[i - 1]

    def __getitem__(self, item):
        demo_idx = bisect.bisect_right(self.demo_lens, item)

        with h5py.File(demo.joinpath('states.data'), 'r') as f:
            if demo_idx > 0:
                step_idx = item - self.demo_lens[demo_idx - 1]
            else:
                step_idx = item
            
            img = torchvision.io.read_image(self.demos[demo_idx].joinpath(f'imgs/img{step_idx}.png')).float() / 255

            sentence = gen_sentence(f['gen_attrs'], f['objectives']['0']['targets'])
        
        raise Exception(f'Failed to open file {str(demo.joinpath('states.data'))}')
