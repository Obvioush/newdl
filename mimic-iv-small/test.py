import os

LocList = []
pidLocMap = {}
infd = open(os.path.join('/Users/masaka/Documents/CAMP-master/mimic3/mimic', 'admissions.csv'), 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    pidLocMap = tokens[7]
    if pidLocMap not in LocList:
        LocList.append(pidLocMap)