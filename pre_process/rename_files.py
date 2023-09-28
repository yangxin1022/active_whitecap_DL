import os
import re
from natsort import natsorted

path = 'C:/Users/yang.xin1022/Desktop/cropped_4k'

fileList = os.listdir(path)

# delete all '._' files
for i in fileList:
    if re.match('._',i):
        fname = i
        print(fname)
        os.remove(f'{path}/{fname}')

# get the list of the files again
fileList = os.listdir(path)
# You have to use natsorted 
# to get the same order in the folder
fileList = natsorted(fileList)
print(fileList)
for n, i in enumerate(fileList):
    oldname = path+os.sep+i
    newname = path+os.sep+'raw_'+str(n+1)+'.png'
    os.rename(oldname, newname)
    print(oldname,'======>',newname)