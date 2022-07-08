import matplotlib
import csv
import numpy as np
import sklearn.preprocessing as prep
import torch
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
indices_to_standardize = [] # tracks the indices of the rows with data to standardize with z-score (initially float data)
all_indices_tracked = False

for row in csvreader:
    transformed_data = [] # initialize variable (will be replaced by tensor then concatenated)

    # TODO: tranpose the one hot encoding and concat with the other data so we add dimensions to the input but end up with a tensor of floats
    # then standardize the floats after the fact
    
    row = row[2:] # remove unhelpful label data
    
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
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    ninhibiting = num_to_tensor(row[5])
    transformed_data = torch.cat((transformed_data, ninhibiting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    exciting = num_to_tensor(row[6])
    transformed_data = torch.cat((transformed_data, exciting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    inhibiting = num_to_tensor(row[7])
    transformed_data = torch.cat((transformed_data, inhibiting),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    excited = num_to_tensor(row[8])
    transformed_data = torch.cat((transformed_data, excited),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    inhibited = num_to_tensor(row[9])
    transformed_data = torch.cat((transformed_data, inhibited),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    fireRate = num_to_tensor(row[10])
    transformed_data = torch.cat((transformed_data, fireRate),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    tot_fireRate = num_to_tensor(row[11])
    transformed_data = torch.cat((transformed_data, tot_fireRate),0)
    if not all_indices_tracked: indices_to_standardize.append(transformed_data.shape[0])
    
    # cellType -> OHE
    cell_ohe = convert_to_one_hot(row[12], 3, ('i','p','n')) 
    transformed_data = torch.cat((transformed_data, torch.transpose(cell_ohe,0,1)),0)
    
    all_indices_tracked = True
    rows.append(transformed_data)

input_data = torch.hstack(rows)

print(input_data.shape)
print(indices_to_standardize)


# data_tensor = torch.tensor(rows)

# print(header)
# print(data_tensor.shape)

cell_data.close()

