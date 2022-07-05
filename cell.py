import matplotlib
import csv
import numpy as np
import torch
import torch.nn.functional as F

"""
Hc3-cell labels:

id integer, -- Id used to match original row number in MatLab PyrIntMap.Map matrix

topdir string, -- top level directory containing data

animal string, -- name of animal

ele integer, -- electrode number

clu integer, -- ID # in cluster files

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
		- n: an int representing the number of classes being mapped total across *all* data
		- categories: tuple conatining all categories of data, whose indices can be used to map the excoding to ints for the one_hot func
					e.g. ('data1', 'data2', 'data3', ...)
	outputs:
		- a size n PyTorch tensor containing a list of all zeroes and a single 1 which maps the input to its distinct category
	
	the purpose of this function is to be called at each iteration of reading the raw csv data 
	(or iterating through the rows of data later) and return the corresponding binary one hot encoding 
	"""
	if not categories:	
		# if no mapping is provided, the data is ordered ints (therefore the mapping is inferred)
		if not type(val) == int:
			raise Exception("Mapping cannot be inferred for string data. Please provide categories as well.")
		return F.one_hot(torch.tensor([val]), num_classes=n)
	else:
		# if category is a string, convert to an int then one hot encode
		if not val in categories:
			raise Exception("Invalid category entered, cannot one-hot encode")

		if len(categories) != n:
			raise Exception("Category list must be of size n")

		int_encoding = categories.index(val)
		return F.one_hot(torch.tensor([int_encoding]), num_classes=n)

print(convert_to_one_hot('c', 5, ('test','a','b','c','d')))


# feature representation: 
# animal -> one hot encoding (categories but no numerical ordering); 11 distinct animals
# ele -> one hot encoding; 8
# clu -> num of types of spikes found (keep as int)
# region -> one hot encoding; 9 regions
# all cell number data -> keep as ints since these are counts
# firing rates -> keep as floats
# cellType -> one hot encoding (00, 01, 10 for three cell types)

rows = []
for row in csvreader:
	row = row[2:] # remove unhelpful label data

	# animal (string) -> OHE
	row[0] = convert_to_one_hot(row[0], 11, ('ec012','ec013','ec014','ec016','f01_m','g01_m','gor01','i01_m','j01_m','pin01','vvp01'))
	# electrode (int) -> OHE
	row[1] = convert_to_one_hot(int(row[1]), 8)
	# region (string) -> OHE
	row[3] = convert_to_one_hot(row[3], 9, ('EC2','EC3','EC4','EC5','EC?','CA1','CA3','DG','Unknown'))
	# cellType -> OHE
	row[12] = convert_to_one_hot(row[12], 3, ('i','p','n'))
	print(row)
	rows.append(row)
	break

rows = np.array(rows)



cell_data.close()

