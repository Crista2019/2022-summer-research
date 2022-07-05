import matplotlib
import csv

"""

Hc3-session labels:

id, -- matches row in original MatLab Beh matrix

topdir, -- directory in data set containing data (tar.gz) files

session, -- individual session name (corresponds to name of tar.gz file having data)

behavior, -- behavior, one of: Mwheel, Open, Tmaze, Zigzag, bigSquare, bigSquarePlus, circle, linear, linearOne, linearTwo, midSquare, plus, sleep, wheel, wheel_home

familiarity, -- number of times animal has done task

duration -- recording length in seconds

"""

session_data = open('hc3-session.csv')


csvreader = csv.reader(session_data)

rows = []
for row in csvreader:
	rows.append(row)
	print(row)
	break