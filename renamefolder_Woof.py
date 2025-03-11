import os

import torchvision
import torch
ToImagenet = {'00000':'n02096294',  
            '00001':'n02093754', 
            '00002':'n02111889',  
            '00003':'n02088364',  
            '00004':'n02086240', 
            '00005':'n02089973', 
            '00006':'n02087394', 
            '00007':'n02115641',  
            '00008':'n02099601', 
            '00009':'n02105641'}
Resize128 = torchvision.transforms.Resize(128)

basedir = './Woof_IPC250_1/'
for fn in os.listdir(basedir):
    DSfolder = os.path.join(basedir,fn, 'ipc_250')
    Savefolder = os.path.join(basedir,fn, 'ipc_250_Renamed')
    if not os.path.exists(Savefolder):
        os.mkdir(Savefolder)
    Fileslst = os.listdir(DSfolder)
    for files in Fileslst:
        Class_Folder = os.path.join(DSfolder, files)
        ImagenmLst = os.listdir(Class_Folder)
        # breakpoint()
        LstImg=[]
        SavedImg = torch.zeros(3, 256, 256)
        savedidx = 0
        idx = 0
        for Imagenm in ImagenmLst:
            ImagePath = os.path.join(Class_Folder, Imagenm)
            Img = torchvision.io.read_image(ImagePath)
            ##################################################################################
            idx +=1
            LstImg.append(Img)
            if idx ==4:
                SavedImg[:, 0:128, 0:128] = Resize128(LstImg[0])
                SavedImg[:, 128:256, 0:128] = Resize128(LstImg[1])
                SavedImg[:, 0:128, 128:256] = Resize128(LstImg[2])
                SavedImg[:, 128:256, 128:256] = Resize128(LstImg[3])
                Name = ToImagenet[files]
                SaveClassFloder = os.path.join(Savefolder, Name)
                if not os.path.exists(SaveClassFloder):
                    os.mkdir(SaveClassFloder)
                torchvision.io.write_png(input=SavedImg.type(torch.uint8) , filename=os.path.join(SaveClassFloder, str(savedidx+50)+".png"), compression_level = 0)
                savedidx += 1
                SavedImg = torch.zeros(3, 256, 256)
                LstImg = []
                idx=0

