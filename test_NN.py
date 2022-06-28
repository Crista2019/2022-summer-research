import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# import data from Lars

dummy_bkg_path = 'set3/bkg.pkl' # background data
dummy_sig_path = 'set3/sig.pkl' # signal data
file = open(dummy_bkg_path, 'rb')
bkg_data = pickle.load(file)
file.close()

file = open(dummy_sig_path, 'rb')
sig_data = pickle.load(file)
file.close()

# print('test show data')
# print(sig_data[0])
# print(bkg_data)

# dimensions of data:
# n = 50000 # data points
# m = 2 # x,y coordinate position

# task: correctly classify the difference between signal and background data

x = []
y = []

for i in range(len(bkg_data)):
	x.append(bkg_data[i])
	y.append(False) # label for background is now False

for i in range(len(sig_data)):
	x.append(sig_data[i])
	y.append(True) # label for background is now True

x = np.array(x) # reduce speed of constructing tensor by making data a np array
y = np.array(y)


# wrapping dataset

class dataset(Dataset):
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float)
		self.y = torch.tensor(y, dtype=torch.bool)
		self.length = self.x.shape[0]
	def __getitem__(self,idx):
		return self.x[idx],self.y[idx]
	def __len__(self):
		return self.length

# trainset = dataset(x,y)
# trainloader = DataLoader(trainset, batch_size=64,shuffle=True)

# defining the model class

class Net(nn.Module):
	def __init__(self, input_shape):
		super(Net, self).__init__()
		# define activation functions
		print(input_shape)
		self.fc1 = nn.Linear(input_shape, 32)
		self.fc2 = nn.Linear(32,64)
		self.fc3 = nn.Linear(64,1)

	def forward(self, x):
		# forward functions
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

# net = Net(x.shape[1])
# print(net)
# x = torch.tensor(x, dtype=torch.float)
# print(x.shape)
# net(x) # forward

# TODO:
# batch training, the go to mini batch training
# do batch training with the data loader later

# hyperparameters

learning_rate = 0.01
epochs = 700

# model, data, optimizer, loss

model = Net(x.shape[1])
x = torch.tensor(x, dtype=torch.float) # features [x1,x2]
y = torch.tensor(y, dtype=torch.bool) # labels True/False
optimizer = rotch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent
loss = nn.BCELoss() # binary cross entropy loss

# training

losses = []
accur = []
# ... tbc