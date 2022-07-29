# import matplotlib
# matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import csv
import numpy as np
import sklearn.preprocessing as prep
import sklearn.model_selection as ms
import sklearn.discriminant_analysis as discrim
import sklearn.metrics as met
from sklearn.utils import class_weight
import torch
import torch.utils.data as data_utils
import math
import torch.nn.functional as F

"""
Hc3-cell labels:

id integer, -- Id used to match original row number in MatLab PyrIntMap.Map matrix

topdir string, -- top level directory containing data

animal string, -- name of animal

ele integer, -- electrode number

clu integer, -- ID # in cluster files: result of spike sorting steps -> spikes most likely generated by the same neuron placed into a category (cluster), which is assigned a non-negative integer cluster number 

region string, -- brain region

nexciting integer, -- number of cells this cell monosynaptically excited

ninhibiting integer, -- number of cells this cell monosynaptically inhibited

exciting integer, -- physiologically identified exciting cells based on CCG analysis

inhibiting integer, -- physiologically identified inhibiting cells based on CCG analysis (Detailed method can be found in Mizuseki Sirota Pastalkova and Buzsaki., 2009 Neuron paper.) 

excited integer, -- based on cross-correlogram analysis, the cell is monosynaptically excited by other cells

inhibited integer, -- based on cross-correlogram analysis, the cell is monosynaptically inhibited by other cells

fireRate real, -- meanISI=mean(bootstrp(100,'mean',ISI)); fireRate = SampleRate/MeanISI; ISI is interspike intervals.

totalFireRate real, -- num of spikes divided by total recording length for a period with a high response rate

cellType string -- ''p'=pyramidal, 'i'=interneuron, 'n'=not identified as pyramidal or interneuron

"""

cell_data = open('hc3-cell.csv')

header = ['animal', 'ele', 'clu', 'region', 'nexciting', 'ninhibiting', 'exciting', 'inhibiting', 'excited', 'inhibited', 'fireRate', 'totalFireRate', 'cellType']

csvreader = csv.reader(cell_data)

def convert_to_one_hot(val, n, categories=None):
	"""
	inputs:
		- val: a single string representing ONE element of categorical data to be converted
		- n: an int representing the number of classes being mapped total across *all* data (this can be larger than the categories size, if you want to pad the encoding with zeros)
		- categories: tuple conatining all categories of data, whose indices can be used to map the excoding to ints for the one_hot func
					e.g. ('data1', 'data2', 'data3', ...)
	outputs:
		- a size n PyTorch tensor containing a list of all 0. (floats) and a single 1. which maps the input to its distinct category
	
	the purpose of this function is to be called at each iteration of reading the raw csv data 
	(or iterating through the rows of data later) and return the corresponding binary one hot encoding 
	"""
	if not categories:	
		# if no mapping is provided, the data is ordered ints (therefore the mapping is inferred)
		if not type(val) == int:
			raise Exception("Mapping cannot be inferred for string data. Please provide categories as well.")
		return F.one_hot(torch.tensor([val]), num_classes=n).float()
	else:
		# if category is a string, convert to an int then one hot encode
		if not val in categories:
			raise Exception("Invalid category entered, cannot one-hot encode")

		if len(categories) > n:
			raise Exception("Category list must be of size n or smaller")

		int_encoding = categories.index(val)
		return F.one_hot(torch.tensor([int_encoding]), num_classes=n).float()

def num_to_tensor(num):
    """
    simple function that takes in an string that represents an int or float and returns a size 1 tensor of floats
    """
    return torch.tensor([[float(num)]], dtype=float)

# feature representation: 
# 0: animal -> one hot encoding (categories but no numerical ordering); 11 distinct animals
# 1: ele -> one hot encoding; 16
# 2: clu -> num of types of spikes found (keep as int)
# 3: region -> one hot encoding; 9 regions
# 4, 5, 6, 7, 8, 9: all cell number data -> keep as ints since these are counts
# 10, 11: firing rates -> keep as floats, but use z score
# 12: cellType -> one hot encoding (00, 01, 10 for three cell types)

rows = []
output = []
indices_to_standardize = [] # tracks the indices of the rows with data to standardize with z-score (initially float data)
all_indices_tracked = False

for row in csvreader:
    transformed_data = [] # initialize variable (will be replaced by tensor then concatenated)
    label = []

    row = row[2:] # remove unhelpful label data

    if 'NaN' in row:
        # filter out rows with incomplete data
        continue
    
    # animal (string) -> OHE
    animal_ohe = convert_to_one_hot(row[0], 11, ('ec012','ec013','ec014','ec016','f01_m','g01_m','gor01','i01_m','j01_m','pin01','vvp01'))
    transformed_data = torch.transpose(animal_ohe, 0, 1)

    # electrode (int) -> OHE
    electrode_ohe = convert_to_one_hot(int(row[1])-1, 16)
    transformed_data = torch.cat((transformed_data, torch.transpose(electrode_ohe,0,1)),0)
    
    # CLU -> OHE
    cluster_num = convert_to_one_hot(int(row[2])-1, 32)
    transformed_data = torch.cat((transformed_data, torch.transpose(cluster_num,0,1)),0)

    # region (string) -> OHE
    region_ohe = convert_to_one_hot(row[3], 9, ('EC2','EC3','EC4','EC5','EC?','CA1','CA3','DG','Unknown')) 
    transformed_data = torch.cat((transformed_data, torch.transpose(region_ohe,0,1)),0)

    # cell number values -> floats
    nexciting = num_to_tensor(row[4]) 
    transformed_data = torch.cat((transformed_data, nexciting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    ninhibiting = num_to_tensor(row[5])
    transformed_data = torch.cat((transformed_data, ninhibiting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    exciting = num_to_tensor(row[6])
    transformed_data = torch.cat((transformed_data, exciting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    inhibiting = num_to_tensor(row[7])
    transformed_data = torch.cat((transformed_data, inhibiting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    excited = num_to_tensor(row[8])
    transformed_data = torch.cat((transformed_data, excited),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    inhibited = num_to_tensor(row[9])
    transformed_data = torch.cat((transformed_data, inhibited),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    fireRate = num_to_tensor(row[10])
    transformed_data = torch.cat((transformed_data, fireRate),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    tot_fireRate = num_to_tensor(row[11])
    transformed_data = torch.cat((transformed_data, tot_fireRate),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0]-1)
    
    # cellType -> OHE
    cell_ohe = convert_to_one_hot(row[12], 3, ('i','p','n'))
    label = torch.transpose(cell_ohe,0,1)

    all_indices_tracked = True
    rows.append(transformed_data)
    output.append(label)

cell_data.close()

input_data = torch.hstack(rows) 

# classification for cell type
output_label = torch.hstack(output).float().transpose(-1,0) # each row contains the one hot encoding for the type of cell per subject

# print('interneuron count:', sum(output_label.transpose(-1,0)[-3].numpy(),1))
# print('pyrimidal count:', sum(output_label.transpose(-1,0)[-2].numpy(),1))
# print('not defined:', sum(output_label.transpose(-1,0)[-1].numpy(),1))

# interneuron count: 1133.0
# pyrimidal count: 6096.0
# not defined: 495.0

# iterate over the float (not one hot encoded data) and standardize using z-score
# z = (x - u) / s
for i in indices_to_standardize:

    # convert to numpy array to perform transformation
    altered_row = input_data[i].numpy().reshape(1,-1).transpose()

    # standard scalar in order to convert to z score
    scale = prep.StandardScaler()
    scale.fit(altered_row)
    altered_row = scale.transform(altered_row).transpose()

    # redefine the original data as a tensor of the z-score data
    input_data[i] = torch.tensor(altered_row, dtype=float)

input_data = input_data.transpose(-1,0) # where each row is a different subject and each column is an input feature

# divide into 70% train and 30% train
x_train, x_test, y_train, y_test = ms.train_test_split(input_data, output_label, test_size=0.3, shuffle=False)

# classweights
train_y_ints = np.argmax(y_train,axis=1).numpy()
train_classes = np.unique(train_y_ints)
test_y_ints = np.argmax(y_test,axis=1).numpy()
test_classes = np.unique(test_y_ints)

train_weights = class_weight.compute_class_weight('balanced', classes=train_classes, y=train_y_ints)
train_sampler = data_utils.WeightedRandomSampler(train_weights, train_weights.shape[0], replacement=True)
# combine the train data set
train_data = []
for i in range(len(x_train)):
    train_data.append([x_train[i], y_train[i]])

test_weights = class_weight.compute_class_weight('balanced', classes=test_classes, y=test_y_ints)
test_sampler = data_utils.WeightedRandomSampler(test_weights, test_weights.shape[0], replacement=True)
# combine the test data set
test_data = []
for i in range(len(x_test)):
    test_data.append([x_test[i], y_test[i]])

train_loader = data_utils.DataLoader(train_data, batch_size=20, shuffle=False, sampler=train_sampler)
test_loader = data_utils.DataLoader(test_data, batch_size=20, shuffle=False, sampler=test_sampler)

# start of the neural network

input_dims = input_data.shape[1]
output_dims = output_label.shape[1]
# print(input_data)
# print(output_label)
# print(input_dims, output_dims) # 76, 3
# print(input_data.dtype)
# print(output_label.dtype)

model = torch.nn.Sequential(
    torch.nn.Linear(input_dims,100),
    torch.nn.ReLU(),
    torch.nn.Linear(100,100),
    torch.nn.ReLU(),
    torch.nn.Linear(100,output_dims),
    torch.nn.Softmax(dim=1) 
)


train_losses = []
eval_losses = []

# training step
# model.train()

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc = torch.nn.Linear(input_dims,100)
        self.fc1 = torch.nn.Linear(100,100)
        self.fc2 = torch.nn.Linear(100, output_dims)
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self,y):
        y = F.relu(self.fc(y))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.soft(y)
        return y

models = Network()
if torch.cuda.is_available():
    models = models.cuda()

# hyperparams
learning_rate = 1e-3
loss_fn = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(models.parameters(), lr=learning_rate, momentum=.9)
epoch = 5

for i in range(epoch):
    # train
    epoch_train = []
    models.train()
    for idx, batch in enumerate(train_loader):
        data, label = batch
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        targets = models(data.float())
        loss = loss_fn(targets,label)
        loss.backward()
        optimizer.step()
        epoch_train.append(loss.item())
    train_losses.append(epoch_train)
        # print('train loss',loss.item())
    # test
    epoch_test = []
    models.eval()
    for idx, batch in enumerate(test_loader):
        data, label = batch
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        targets = models(data.float())
        loss = loss_fn(targets,label)
        epoch_test.append(loss.item())
    # eval_losses.append(loss.item()*data.size(0))
    eval_losses.append(epoch_test)
        # print('test loss',loss.item())

# plot the losses
fig1, ax1 = plt.subplots()
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
for i in range(epoch):
    ax1.scatter(np.arange(len(train_losses[i])),train_losses[i], label='train loss at epoch '+str(i))
    ax1.scatter(np.arange(len(eval_losses[i])),eval_losses[i], label='evaluation loss at epoch '+str(i))
ax1.legend()
plt.tight_layout()
plt.savefig('disc_hists.png')
plt.show()
"""
def decipher_class(input_data, offset=False):
    # input: a dataset of type float tensor in form: [[0.,0.,1.],[0.00002,1.00,0.008],...]
    # output: a numpy array containing floats of the argmax index of the input: [2,1,...]
    # if offset is set to true, an offset will be applied to avoid overlapping ticks for graphs

    output_data = []
    for row in input_data:
        class_num = np.argmax(row)
        if offset:
            output_data.append(class_num.item()+.1) # will add .1 offset
        else:
            output_data.append(class_num.item())
    return np.array(output_data)

#  testing step
model.eval()
with torch.no_grad():
    out_data = model(x_test.float())

    print(y_test[:][0])

    # discrimination histogram graphing
    fig2, ax2 = plt.subplots(3) # graph all 3 next to each other

    # discrimination plot for class 0: interneurons
    class_0 = out_data[:,0]
    c0_labeled0 = y_test[:,0] == 0
    c0_labeled1 = y_test[:,0] == 1
    c0_labeled2 = y_test[:,0] == 2
    interneurons = class_0[c0_labeled0].detach().numpy().flatten()
    pyramidal = class_0[c0_labeled1].detach().numpy().flatten()
    unlabeled = class_0[c0_labeled2].detach().numpy().flatten()*50


    ax2[0].hist(interneurons, bins=8, range=[0,1], label='interneurons', density=True, alpha=0.5)
    ax2[0].hist(pyramidal, bins=8, range=[0,1], label='pyramidal', density=True, alpha=0.5)
    ax2[0].hist(unlabeled, bins=8, range=[0,1], label='unlabeled', linewidth=1.7, histtype=u'step', density=True, alpha=0.5)
    ax2[0].set_title('Interneuron Discrimination Plot')
    ax2[0].legend(bbox_to_anchor=(1.5,.6))

    # discrimination plot for class 1: pyramidal cells
    class_1 = out_data[:,1]
    c1_labeled0 = y_test[:,1] == 0
    c1_labeled1 = y_test[:,1] == 1
    c1_labeled2 = y_test[:,1] == 2
    interneurons = class_1[c1_labeled0].detach().numpy().flatten()
    pyramidal = class_1[c1_labeled1].detach().numpy().flatten()
    unlabeled = class_1[c1_labeled2].detach().numpy().flatten()*50


    ax2[1].hist(interneurons, bins=8, range=[0,1], label='interneurons', density=True, alpha=0.5)
    ax2[1].hist(pyramidal, bins=8, range=[0,1], label='pyramidal', density=True, alpha=0.5)
    ax2[1].hist(unlabeled, bins=8, range=[0,1], label='unlabeled', linewidth=1.7, histtype=u'step', density=True, alpha=0.5)
    ax2[1].set_title('Pyramidal Cell Discrimination Plot')
    ax2[1].legend(bbox_to_anchor=(1.5,.6))

    # discrimination plot for class 1: unlabeled cells
    class_2 = out_data[:,2]
    c2_labeled0 = y_test[:,2] == 0
    c2_labeled1 = y_test[:,2] == 1
    c2_labeled2 = y_test[:,2] == 2
    interneurons = class_2[c2_labeled0].detach().numpy().flatten()
    pyramidal = class_2[c2_labeled1].detach().numpy().flatten()
    unlabeled = class_2[c2_labeled2].detach().numpy().flatten()*50
    print(interneurons)
    print(pyramidal)
    print(unlabeled)


    ax2[2].hist(interneurons, bins=8, range=[0,1], label='interneurons', density=True, alpha=0.5)
    ax2[2].hist(pyramidal, bins=8, range=[0,1], label='pyramidal', density=True, alpha=0.5)
    ax2[2].hist(unlabeled, bins=8, range=[0,1], label='unlabeled', linewidth=1.7, histtype=u'step', density=True, alpha=0.5)
    ax2[2].set_title('Unlabeled Cell Discrimination Plot')
    ax2[2].legend(bbox_to_anchor=(1.5,.6))

    plt.tight_layout()
    plt.show()
    # plt.savefig('disc_hists.png')

    # analysis that requires the labels to be one node (i.e. the index of the one hot encoded class) rather than 3
    predicted_class_w_offset = decipher_class(out_data, True)
    predicted_class = decipher_class(out_data)
    expected_class = decipher_class(y_test)

    # test accuracy
    print('BALANCED ACCURACY:',met.balanced_accuracy_score(expected_class,predicted_class))

    correct_x = []
    correct_y_w_offset = []
    wrong_x = []
    wrong_y_w_offset = []
    for i in range(predicted_class.shape[0]):
        # account for the offset!
        if predicted_class[i] == expected_class[i]:
            correct_x.append(i)
            correct_y_w_offset.append(predicted_class_w_offset[i])
        else:
            wrong_x.append(i)
            wrong_y_w_offset.append(predicted_class_w_offset[i])

    # raw, unbalanced accuracy
    # accuracy = len(correct_x)/predicted_class.shape[0]
    # print('test accuracy:',accuracy)

    fig3, ax3 = plt.subplots(figsize=(15,4))
    ax3.set_title('Classification Accuracy')
    ax3.set_xlabel('Cell')
    ax3.set_ylabel('Predicted Class')
    ax3.scatter(np.arange(expected_class.shape[0]), expected_class, color='black', marker='|', alpha=.3, label='expected class')
    ax3.scatter(np.array(correct_x), np.array(correct_y_w_offset), color='green', marker='|', alpha=.3, label='correctly classified by model')
    ax3.scatter(np.array(wrong_x), np.array(wrong_y_w_offset), color='red', marker='|', alpha=.3, label='incorrectly classified by model')
    plt.legend(bbox_to_anchor=(.8,.6))
    plt.tight_layout()
    plt.show()
    # plt.savefig()

    # Discriminant Analysis
    X = x_test
    y = expected_class
    target_names = ['interneuron','pyramidal','not identified']

    analysis = discrim.LinearDiscriminantAnalysis()
    data_plot = analysis.fit(X,y).transform(X)

    # create LDA plot
    plt.figure()
    colors = ['red','blue','green']
    lw = 2

    # plotting
    for color, i, target_name in zip(colors, [0,1,2], target_names):
        plt.scatter(data_plot[y==i,0], data_plot[y==i,1],alpha=.8,color=color,label=target_name)
    plt.legend(loc='best',shadow=False,scatterpoints=1)

    plt.show()

"""
