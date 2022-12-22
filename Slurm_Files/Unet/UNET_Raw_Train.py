#%% Import Packages
from Cloud_Loader import CloudDataset
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from UNET_Network import UNET
import time
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torch import cuda, nn


#%% Adjustables

# Number of epochs we are going to run for
epochs = 60

# Minibatch Size
batch_size = 8

# learning rate
learning_rate = 0.0001

# name of the network
netname = 'UNETRaw/UNET_Raw_Weights.pth'

# name of the weights
trendname = 'UNETRaw/UNET_Raw_Weights.csv'


#%% Import Data

# Set the base path and load in the dataset 
base_path = Path('Data/95-cloud_training')

data = CloudDataset(base_path/'train_red', 
                    base_path/'train_green', 
                    base_path/'train_blue', 
                    base_path/'train_nir',
                    base_path/'train_gt')


# Split into training and testing data
train_ds , valid_ds , test_ds = torch.utils.data.random_split(data, [ 0.6 , 0.2 , 0.2 ], generator=torch.Generator().manual_seed(42))

# Load training, validation, and testing data
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds , batch_size = batch_size, shuffle = True)

print(f' training dataset length: {len(train_ds)}')
print(f' validation dataset length: {len(valid_ds)}')
print(f' training dataset length: {len(test_ds)}')

#%% Load Network

unet = UNET(4,2) # 4 channels in (R/G/B/NIR) with 2 channels out (Cloud or no)

# unet.load_state_dict(torch.load('../input/weights/d95_e80_aug.pt'))
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f' Running on {dev} ')


#%% Define Training Regime

def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.to(dev) # send the model to the device if it is available

    train_loss, valid_loss, train_acc ,   valid_acc  = [] , [] , [] , [] # look at training and validation los

    best_acc = 0.0 # set best accuracy

    for epoch in range(epochs): # for the specified number of epochs
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'valid']:
            if phase == 'train': 
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl # load in the training datasets
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl # load in the validation datasets

            # set losses and accuracies to 0 
            running_loss = 0.0 
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.to(dev)
                y = y.to(dev)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size 

                if step % 100 == 0:
                    clear_output(wait=True)

            # loss and accuracy by the epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)
            # epoch_acc = round(epoch_acc*100 , 3 )

            
            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            # Store the stats in a string of tensors
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)
            train_acc.append(epoch_acc) if phase=='train' else valid_acc.append(epoch_acc)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    torch.save(model.state_dict(), netname)

    
    return train_loss, valid_loss , train_acc , valid_acc    

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.to(dev)).float().mean()


#%% Run the Network

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(unet.parameters(), lr= learning_rate )

train_loss, valid_loss, train_acc , valid_acc = train(unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs = epochs)


#%%  Save the loss and accuracy progression

# Training accuracy
t_acc = np.zeros(len(train_acc))
for i , r in enumerate(train_acc):
    t_acc[i] = np.asarray(train_acc[i].cpu())

# validation accuracy
v_acc = np.zeros(len(valid_acc))
for i , r in enumerate(valid_acc):
    v_acc[i] = np.asarray(valid_acc[i].cpu())

# Training Loss
t_loss = np.zeros(len(train_loss))
for i , r in enumerate(train_loss):
    t_loss[i] = np.asarray(train_loss[i].detach().cpu())
    
# Validation Loss
v_loss = np.zeros(len(valid_loss))
for i , r in enumerate(valid_loss):
    v_loss[i] = np.asarray(valid_loss[i].cpu())

# Save Results in dictionary
dict = {
    'Testing Accuracy' : t_acc,
    'Validation Accuracy' : v_acc,
    'Testing Loss' : t_loss,
    'Validation Loss' : v_loss,
}

df = pd.DataFrame(dict)
df.to_csv(trendname)