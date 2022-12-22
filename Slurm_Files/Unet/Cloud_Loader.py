#%% Import packages
from pathlib import Path
import torch
import numpy as np
import numpy as np
from pathlib import Path
from PIL import Image
import torch

#%% Define how we load the dataset
class CloudDataset():
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()] 
        self.pytorch = pytorch # use pytorch (?)
    
    # combine the files for red, green, blue, ground truth, and infra red directories 
    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):
        
        files = {'red': r_file, 
                'green':g_dir/r_file.name.replace('red', 'green'),
                'blue': b_dir/r_file.name.replace('red', 'blue'), 
                'nir': nir_dir/r_file.name.replace('red', 'nir'),
                'gt': gt_dir/r_file.name.replace('red', 'gt')}
        
        return files
                                    
    def __len__(self):
            return len(self.files) # get the length of the dataset
    
    # open as an array 
    def open_as_array(self, idx, invert=False, include_nir=False, false_color_aug=False):
        # Create a stacked RGB image
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                        ])
        
        # Create a false color image if you want to 
        if (false_color_aug):
            indexes = np.arange(3)
            np.random.shuffle(indexes)
            raw_rgb = np.stack([raw_rgb[indexes[0]],
                                raw_rgb[indexes[1]],
                                raw_rgb[indexes[2]],
                            ], axis=2)

        else: # otherwise, just use the raw rgb
            raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                    np.array(Image.open(self.files[idx]['green'])),
                    np.array(Image.open(self.files[idx]['blue'])),
                ], axis=2)

        # if you are including the NIR, expand the dimensions and concatonate
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        # trannspose for NIR/RGB Image
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
    
        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    
    # open the masks
    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    

    def __getitem__(self, idx):
        
        x = torch.tensor(self.open_as_array(idx,
                                    invert=self.pytorch,
                                    include_nir=True,
                                    false_color_aug=True),
                                    dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False),
                                  dtype=torch.torch.int64)
        
        return x, y
    
    def open_as_pil(self, idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s
