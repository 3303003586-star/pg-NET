# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:28:13 2024

@author: Stefanos.Bakirtzis
"""
import cv2
import copy 
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colrs
import matplotlib.pyplot as plt

from skimage.io import imread
plt.close("all")

#%% --------------------%  (1) Select Sampme  %--------------------
b = 1
ant = 1
f = 1
sp = 20

freqs_GHz = [0.868, 1.8, 3.5]

#%% --------------------%  (2) Get Position and Geometry Details %--------------------
Sampling_positions = pd.read_csv("../dataset/Positions/Positions_B" + str(b) + "_Ant" + str(ant) + "_f" + str(f) + '.csv')
Building_Details = pd.read_csv("../dataset/Building_Details/B" + str(b) + "_Details.csv")

# 建筑宽高
W, H = Building_Details["W"].iloc[0], Building_Details["H"].iloc[0]

# 行等差排列与列等差排列，用于生成天线辐射图
X_points = np.repeat(np.linspace(0, W-1, W), H, axis=0).reshape(W, H).transpose()
Y_points = np.repeat(np.linspace(0, H-1, H), W, axis=0).reshape(H, W)

# 天线位置
x_ant = Sampling_positions["Y"].loc[sp]
y_ant = Sampling_positions["X"].loc[sp]

# print(Sampling_positions.loc[sp])
# print(Building_Details["W"].iloc[0])
# print(Building_Details["H"].iloc[0])

# 输入输出数据
y_PL = np.array(imread("../dataset/Outputs/Task_3_ICASSP/B" + str(b) + "_Ant" + str(ant) + "_f" + str(f) + "_S" + str(sp) + '.png'))
X_input = np.array(imread(("../dataset/Inputs/Task_3_ICASSP/B" + str(b) + "_Ant"+ str(ant) + "_f" + str(f) + "_S" + str(sp) + '.png')))

Antenna_Azimuth_Pattern = np.genfromtxt("../dataset/Radiation_Patterns/Ant" + str(ant) + "_Pattern.csv", delimiter=',', skip_header=1)
print(Antenna_Azimuth_Pattern)
    
Angles = -(180/np.pi) * np.arctan2((y_ant - Y_points), (x_ant - X_points)) + 180 + Sampling_positions['Azimuth'].iloc[sp]
print(Angles.shape)
Angles = np.where(Angles > 359, Angles - 360, Angles).astype(int)
print(Angles.shape)
G = Antenna_Azimuth_Pattern[Angles]
print(G.shape)

freq = freqs_GHz[f-1]
lamda = 0.3/(freq) 

my_cmap = copy.copy(cm.get_cmap('jet_r'))

plt.figure()
plt.imshow(X_input[:,:,0])

plt.figure()
plt.imshow(X_input[:,:,1])

plt.figure()
plt.imshow(X_input[:,:,2])


plt.figure()
plt.imshow(Angles)

plt.figure()
plt.imshow(G)

plt.figure()
plt.imshow(y_PL[:, :], cmap = my_cmap)

plt.show()

#%% --------------------%  (4) Resize %--------------------
plt.close("all")
size = 512

X_input_size = np.zeros([size, size, 3])
a, b = y_PL.shape
y_PL_sized = cv2.resize(y_PL, (size, size), interpolation=cv2.INTER_NEAREST)
y_PL_sized1 = cv2.resize(y_PL_sized, (b, a), interpolation=cv2.INTER_NEAREST)

mae = abs(y_PL_sized1 - y_PL)
print(mae)
count = np.count_nonzero(mae == 0)
print(count)

X_input_size = cv2.resize(X_input,  (size, size), interpolation=cv2.INTER_NEAREST)

plt.figure()
plt.imshow(X_input_size[:,:,0])

plt.figure()
plt.imshow(X_input_size[:,:,1])

plt.figure()
plt.imshow(X_input_size[:,:,2])

plt.figure()
plt.imshow(y_PL_sized1[:,:], cmap = my_cmap)

plt.figure()
plt.imshow(y_PL[:,:], cmap = my_cmap)
   

#%% --------------------%  (5) Indicative data augementation %--------------------   
plt.close("all")
y_PL_flip =  np.flip(y_PL)   
X_input_flip =  np.flip(X_input)   


plt.figure()
plt.imshow(X_input_flip[:,:,0])

plt.figure()
plt.imshow(X_input_flip[:,:,1])

plt.figure()
plt.imshow(X_input_flip[:,:,2])

plt.figure()
plt.imshow(y_PL_flip[:,:], cmap = my_cmap)


#%% --------------------%  (6h) Indicative data loader %--------------------   
 
import torch
from torch.utils.data import Dataset, DataLoader


class DataGenerator(Dataset):
    """Creates Dataset and Picks up a SINGLE Random Sample from Dataset"""

    def __init__(self, device, list_IDs, file_names, input_path, output_path, 
                 dim_X, dim_y, n_channels = 3, 
                 batch_size=1, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator, 0 to Dataset_size-1
        :param file_names: list of file names
        :param input_path: path to input data
        :param output_path: path to PL radio maps
        :param batch_size: The generator picks up a SINGLE sample, the batch size is defined by the DataLoader
        :param dim_X: input dimensions to resize
        :param dim_y: ouput dimensions to resize 
        :param n_channels: number of input channels; default is 3, but change if you use more features
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.file_names = file_names
        self.input_path = input_path
        self.output_path = output_path
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.batch_size = batch_size 
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.device = device

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch and find IDs
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Read Input and PL images 
        X = self._generate_X(list_IDs_temp) 
        y = self._generate_y(list_IDs_temp)
        
        return torch.from_numpy(X).to(self.device ),  torch.from_numpy(y).to(self.device )
       
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

 
    def _generate_X(self, list_IDs_temp):

        X =   imread(self.input_path + self.file_names[list_IDs_temp[0]] + ".png")
        #   X = np.moveaxis(X, -1, 1)    , if you want channels first                           

        return  np.squeeze( X.astype(np.float32) )
  
    def _generate_y(self, list_IDs_temp):
     
        y =  imread(self.output_path + self.file_names[list_IDs_temp[0]]  + ".png")   

        return np.squeeze( y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # change to  
input_path =  "Inputs/Task_1_ICASSP/" 
target_path = "Outputs/Task_1_ICASSP/" 


Input_list_file_names = []

for b in range(1, 25+1): 
   for s in range (50): 
    Input_list_file_names.append("B" + str(b) +  "_Ant"+  str(1) + "_f"  + str(1) + "_S" +str(s))   


Input_list_IDs =  np.arange(0, len(Input_list_file_names), 1, dtype=int)
train_generator = DataGenerator (device, Input_list_IDs, Input_list_file_names, input_path, target_path, 
                 dim_X = (512,512), dim_y = (512,512) ,
                 n_channels=3, batch_size=1,  
                 shuffle=True)

batch_size =1 # if you want to use a batch_size different thatn 1, you should resize the data when loaded in the DataGenerator, e.g., in the _generateX function
              # otheriwse, samples with different size will end up in the same batch and the generator will give an error
train_loader = DataLoader(train_generator, batch_size=batch_size, shuffle=True)

import matplotlib.pyplot as plt

import copy 
import matplotlib.cm as cm
import matplotlib.colors as colrs

my_cmap = copy.copy(cm.get_cmap('jet_r'))

 
a = train_generator.__getitem__(0)

input_ = a[0].cpu().detach().numpy()
output_ = a[1].cpu().detach().numpy()

# Check Generator is working
sample_id =0



plt.close("all")
plt.figure(1)
plt.imshow(np.squeeze( input_[:,:,0] )  ) 

plt.figure(2)
plt.imshow(np.squeeze( input_[:,:,1] )  ) 

plt.figure(3)
plt.imshow(np.squeeze( input_[:,:,2] )  ) 


plt.figure(5)
plt.imshow((  np.squeeze(  ( output_[:,:]  ))) , cmap = my_cmap ) 
 
plt.close("all")

# Check Loader is working
epochs = 3
for i in range(epochs):
     
    print("Epcoch: ", i, " entering Trainng")
    for local_batch, local_file_names in (train_loader):
        
        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[0,:,:,0] )  ) 

        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[0,:,:,1] )  ) 

        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[0,:,:,2] )  ) 

        plt.figure()
        plt.imshow((  np.squeeze(  ( local_file_names.cpu().detach().numpy()[0,:,:]) )) , cmap = my_cmap ) 
        """
        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[1,:,:,0] )  ) 

        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[1,:,:,1] )  ) 

        plt.figure()
        plt.imshow(np.squeeze( local_batch.cpu().detach().numpy()[1,:,:,2] )  ) 


        plt.figure()
        plt.imshow((  np.squeeze(  ( local_file_names.cpu().detach().numpy()[1,:,:]))) , cmap = my_cmap ) 
        """
        print(block)