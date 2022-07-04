import matplotlib

cell_data = 'hc3-cell.csv'
session_data = 'hc3-session.csv'

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

-------------------------------------------------------------------------------------------------------------

Hc3-session labels:

id, -- matches row in original MatLab Beh matrix

topdir, -- directory in data set containing data (tar.gz) files

session, -- individual session name (corresponds to name of tar.gz file having data)

behavior, -- behavior, one of: Mwheel, Open, Tmaze, Zigzag, bigSquare, bigSquarePlus, circle, linear, linearOne, linearTwo, midSquare, plus, sleep, wheel, wheel_home

familiarity, -- number of times animal has done task

duration -- recording length in seconds

"""