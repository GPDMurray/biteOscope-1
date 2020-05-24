'''
author: Felix Hol
date: 2020 04 24
content: runs DeepLabCut inference on timelapse images of mosquitoes using model trained on aegypti and ablopictus
Takes cropped frames as input
Output is:
1) body part coordinates per DeepLabCut standards
2) an .avi of the images (this is only used for make the labelled video and/or quick viewing, not for inference)
3) labelled video
modified by Greg Murray 2020 May 24 n.b. activate DLC environment and python in cmd
added static values for videowriter size, problem with - "ScannerError: mapping values are not allowed here
  in "D:\BiteOscope_test_images\config.yaml", line 117, column 73"
'''


import os
os.environ["DLClight"]="True"

import deeplabcut
# from pathlib import Path

import cv2
import numpy as np
import glob
from PIL import Image

print('deeplabcut version: ' + str(deeplabcut.__version__))


import tensorflow as tf
print('tensorflow version: ' + str(tf.__version__))


# tf.test.gpu_device_name()

config = 'D:/BiteOscope_test_images/config.yaml'


dataDir = 'D:/BiteOscope_test_images/test_ouput/testcrops_p1.0/'
files = sorted(glob.glob(dataDir +'*.png'))
# saveDirMovie = 'D:/BiteOscope_test_images/'

img_array = []

for filename in files:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


videoBaseName = os.path.basename(os.path.normpath(dataDir))

if '.' in videoBaseName:
    videoBaseName = videoBaseName[:-2]

videoName = dataDir + videoBaseName + '.avi'
out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), 25, (640,480))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

videoPath = videoName
imageDir = dataDir

deeplabcut.analyze_time_lapse_frames(config, imageDir, save_as_csv=True, rgb=False)

deeplabcut.create_labeled_video(config, [videoPath], filtered=False)