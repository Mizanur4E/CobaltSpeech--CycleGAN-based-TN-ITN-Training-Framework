import os
import fnmatch
import shutil


i=0

tar_dir = '/home/nayan/formatterData/'
with os.scandir('/home/alok/voicescript-data/downloaded-data') as entries:
    for entry in entries:
        with os.scandir(entry) as files:
            for file in files:
                a = os.path.join(file,'Delivery')
                name= str(file)
                name = name[11:-2]
                name= name+'.txt'
                
                if os.path.exists(a) == True : 
                    os.chdir(a)
                    for file_name in os.listdir(a):
                     if fnmatch.fnmatch(file_name, name):
                        i =i+1
                        shutil.copyfile(file_name,tar_dir)
                        print(file_name)
print(i)
