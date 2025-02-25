import cv2 as cv
import numpy as np
from PIL import Image
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import RPI.GPIO as GPIO

#from picamera import PiCamera

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def _init_(self):
        super(SiameseNetwork, self)._init_()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride = 2),

            nn.Conv2d(384, 420, kernel_size = 3,stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,stride = 2),

            nn.Conv2d(420, 512, kernel_size = 3,stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,stride = 2)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1,output2

df = pd.read_csv('/home/pi/class data.csv')
df = df.sort_values(by = 'Name',ignore_index = True)
df['Attendance'] = 0
names = list(df['Name'])

model = torch.load('big_model_2.pth')
model.load_state_dict(torch.load('big_model_2_weights.pth'))
transformation = transforms.Compose([transforms.Resize((150,150)),
				     transforms.ToTensor()])

haar = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#camera = PiCamera.start_progress()
#camera.resolution = (640,480)
#camera.vfilp = True
#time.sleep(2)

#camera.capture('test.jpg')

img_path = os.path.join('/home/pi/Images')
faces,images,face_images = [],[],[]

threshold = 0.8

test_img = cv.imread('/home/pi/test.jpg')
face = haar.detectMultiScale(test_img,scaleFactor = 1.1, minNeighbors = 5)
for i in range(len(face)) :
    sim = []
    for (x,y,w,h) in face[i].reshape((1,4)) :
        face_img = test_img[y:y+h,x:x+w]
        face_img = Image.fromarray(face_img)
        face_img = transformation(face_img)
        face_img = face_img.unsqueeze(0)#
        for img in os.listdir(img_path) :
            im = cv.imread(os.path.join(img_path,img))
            im = Image.fromarray(im)
            im = transformation(im)
            im = im.unsqueeze(0)
            out1,out2 = model(im,face_img)
            euclidean_dist = F.pairwise_distance(out1,out2)
            euclidean_dist = euclidean_dist.detach().numpy()
            sim.append(euclidean_dist)
        sim = np.array(sim)
        idx = int(np.argmin(sim))
        detect = sim[idx]
        if (detect < threshold) :
            df.loc[idx,'Attendance'] = df.loc[idx,'Attendance'] + 1
            with open('class data.csv','w',newline = '') as cls :
                write = csv.writer(cls)
                write.writerow(df.columns)
                for i in range(len(names)) :
                    write.writerow(df.iloc[i])
        else :
            pass
print(df)

