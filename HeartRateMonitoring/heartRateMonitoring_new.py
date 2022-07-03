import matplotlib


from PIL import Image
from PIL import ImageDraw

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.insert(0, './SkinDetector')
import pkg_resources
import random

import numpy as np
import cv2
import dlib

from imutils.video import VideoStream
from imutils import face_utils 
import imutils
import warnings
warnings.filterwarnings("ignore")
import argparse

import time

from matplotlib import cm

framerate = 30 

# FREQUENCY ANALYSIS
nsegments = 12
 
plot =  True
image_show = True


from_webcam = True
camera = cv2.VideoCapture(0)
start = 0
end = 1100
rate = 400
step = 70
meanOfChanells = np.empty((0, 3), float) 

leftRoi = [2, 3, 32, 31,30, 29, 41, 42, 2]
RightRoi = [16, 15, 36, 31, 30, 29, 48,47,16]

start_index = start
end_index = end

# startPoint = ('Nan', 'Nan')
# endPoint = ('Nan', 'Nan')


 # number of final frames
if end_index > 0:
    nb_frames = end_index - start_index


# plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
scale_percent = 80 # percent of original size
width = int(640 * scale_percent / 100 )
height = int(480 * scale_percent / 100 )
dim = (width, height)
hr = 0

# 2a = (1+ (end-rate) / step)
 # loop on video frames
frame_counter = 0
i = start_index + 1
a=0
# meanOfChanellsTotal = np.empty((0,3),float)
meanOfChanellsTotal = []
hr_list = []
signal_list = []
f_max_list = []
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
def animate(i,hr_list):
    
    plt.cla()
    plt.xlim(0,10)
    plt.ylim(50,130)
    #plt.plot(hr_list,label="HeartRate",linestyle=":", linewidth=2)
    plt.plot(hr_list,label="HeartRate")
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(1350,300,530, 530)
    plt.xlabel("Time")
    plt.ylabel("Heart Rate")
    plt.legend()
    
ani = FuncAnimation(plt.gcf(),animate,fargs=(hr_list,),interval = 3000)
plt.tight_layout()
plt.show()   


total_time = time.time()
t_start = time.time()
while (i >= start_index and i <= end_index):
    (grabbed, image) = camera.read()
    if not grabbed:
        print("Yakalanamadı")
        continue
    image2= np.copy(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get faces into webcam's image 
    rects = detector(gray, 0)
    # marks = 0
    if(len(rects) == 0):
        cv2.putText(image, "No Face", (20,20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
    # For each detected face, find the landmark.
    for (z, rect) in enumerate(rects):
        

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        
       
        
        pt1 = shape[RightRoi[0]-1]
        pt2 = shape[RightRoi[1]-1]
        pt3 = shape[RightRoi[2]-1]
        pt4 = shape[RightRoi[3]-1]
        pt5 = shape[RightRoi[4]-1] 
        pt6 = shape[RightRoi[5]-1]
        pt7 = shape[RightRoi[6]-1]
        pt8 = shape[RightRoi[7]-1]
        pt9 = shape[RightRoi[8]-1]


        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners1 = np.array([[pt1, pt2, pt3, pt4, pt5,pt6,pt7,pt8]], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners1, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
      

        pt10 = shape[leftRoi[0]-1]
        pt11 = shape[leftRoi[1]-1]
        pt12 = shape[leftRoi[2]-1]
        pt13 = shape[leftRoi[3]-1]
        pt14 = shape[leftRoi[4]-1]
        pt15 = shape[leftRoi[5]-1]
        pt16 = shape[leftRoi[6]-1]
        pt17 = shape[leftRoi[7]-1]
        pt18 = shape[leftRoi[8]-1]
        
        
        roi_corners2 = np.array([[pt10, pt11, pt12,pt13, pt14,pt15,pt16,pt17,pt18 ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners2, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        # maskeleme-bitiş
        
        roi_corners_points = np.array([[pt10,pt11,pt12,pt13,pt3, pt2,pt1,pt8,pt7,pt6,pt16,pt17,pt18 ]])
        roi_corners_points = roi_corners_points[0,:,:]
        ## vertices = np.concatenate((roi_corners1,roi_corners2)) 
        liste = tuple(map(tuple, roi_corners_points))
        
        # filtremelem-başlangıç 
        
        filter_arr = np.logical_and(masked_image[:,:,0] > 5, masked_image[:,:,1]>5, masked_image[:,:,2]>5 )
        MaskedImageWithoutZero = masked_image[filter_arr]
        
        # MaskedImageWithoutZero = masked_image[masked_image > 0]     
        BlueChaneelwithOutZero = MaskedImageWithoutZero[:,0]
        GreenChaneelwithOutZero = MaskedImageWithoutZero[:,1]
        RedChaneelwithOutZero = MaskedImageWithoutZero[:,2]

        meanOfBlue = np.mean(BlueChaneelwithOutZero)
        meanOfGreen = np.mean(GreenChaneelwithOutZero)
        meanOfRed = np.mean(RedChaneelwithOutZero)
        
        
        stdOfBlue = np.std(BlueChaneelwithOutZero)
        stdOfGreen = np.std(GreenChaneelwithOutZero)
        stdOfRed = np.std(RedChaneelwithOutZero)
        
        #
        filter_arr1 = np.logical_and(  (meanOfBlue - stdOfBlue )  <=  MaskedImageWithoutZero[:,0], MaskedImageWithoutZero[:,0]  <=  (meanOfBlue + stdOfBlue )  )
        filter_arr2 = np.logical_and(  (meanOfGreen - stdOfGreen )  <=  MaskedImageWithoutZero[:,1], MaskedImageWithoutZero[:,1]  <=  (meanOfGreen + stdOfGreen )  )
        filter_arr3 = np.logical_and(  (meanOfRed - stdOfRed )  <=  MaskedImageWithoutZero[:,1], MaskedImageWithoutZero[:,1]  <=  (meanOfRed + stdOfRed )  )
        filter_arr4 = np.logical_and(  filter_arr1, filter_arr2, filter_arr3 )

        MaskedImageWithoutZeroAndFiltered = MaskedImageWithoutZero[filter_arr4]
            
        
        meanOfBlueFiltered = np.mean(MaskedImageWithoutZeroAndFiltered[:,0])
        meanOfGreenFiltered = np.mean(MaskedImageWithoutZeroAndFiltered[:,1])
        meanOfRedFiltered = np.mean(MaskedImageWithoutZeroAndFiltered[:,2])
        
        meanOfChanells = np.append(meanOfChanells,[np.array([meanOfRedFiltered,meanOfGreenFiltered,meanOfBlueFiltered])],axis= 0)
        # 400 ye gelip bir öncekine geçmeden listeye ata 
        if( i >= rate and (i - (step * a)) == rate):

            elapsed_time = time.time() - t_start
            framerate = rate/elapsed_time if i == rate else step/elapsed_time
            l = int(framerate * 1.6)
            print("l: {},  elapsed time: {}, framerate: {}".format(l,elapsed_time,framerate))
            meanOfChanellsCal = meanOfChanells[ (a*step) : (rate+(a*step)) , : ]
            H = np.zeros(meanOfChanellsCal.shape[0])
            
            for t in range(0, (meanOfChanellsCal.shape[0] - l + 2 )):
                
                # Step 1: Spatial averaging
                
                C = meanOfChanellsCal[t:t+l-1,:].T
                
                if t == 3:
                    plot = False
            
                
                mean_color = np.mean(C, axis=1)
                 
                 
                diag_mean_color = np.diag(mean_color)
                 
                 
                diag_mean_color_inv = np.linalg.inv(diag_mean_color)
                
                 
                Cn = np.matmul(diag_mean_color_inv,C)
                
             
                 #Step 3: 
                projection_matrix = np.array([[0,1,-1],[-2,1,1]])
                S = np.matmul(projection_matrix,Cn)
                S = projection_matrix@Cn
                
            
                #Step 4:- 2D signal to 1D signal
                std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
                #print("std",std)
                P = np.matmul(std,S)
                #P = std@S
                #print("P",P)
                
            
                #Step 5: Overlap-Adding
                P_std = np.std(P)
                if P_std==0:
                    P_std = 1
                
                H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P) 
            
                
            H = H[np.logical_not(np.isnan(H))]
            signal = H
            signal_list.append(signal)
                 
            
            
            #FFT to find the maxiumum frequency
            # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
            #segment_length = (2*signal.shape[0]) // (nsegments + 1) 
            segment_length = 256
            # print("nperseg",segment_length)
             
            from scipy.signal import welch
            signal = signal.flatten()
            green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=segment_length) #, scaling='spectrum',nfft=2048)
            
            
            
            # green_psd = green_psd.flatten()
            first = np.where(green_f > 0.9)[0] # 0.8 for 300 frames
            last = np.where(green_f < 1.8)[0]
                
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
                
            # print("Range of interest",range_of_interest)
            max_idx = np.argmax(green_psd[range_of_interest])
            f_max = green_f[range_of_interest[max_idx]]
            f_max_list.append(f_max)    
            hr = f_max * 60.0
            hr_list.append(hr)
            text = "HeartRate: {0:5.3f}".format(hr)
            print("Heart rate = {0}".format(hr))
            t_start = time.time()
            a+=1

            
        i += 1 
    
    
    text = "HeartRate: {0:5.3f}".format(hr) if hr !=  0 else "FrameNumber:{}".format(i,)
    # text2 = "Max(fr): {0:5.3f}".format(f_max) if hr !=  0 else "Max(fr): ..."
        
    image2 = Image.fromarray(image2)
    poly = Image.new('RGBA', image2.size)
    pdraw = ImageDraw.Draw(poly)
    pdraw.polygon(liste,fill=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255),64),outline=(0,0,0,255))
    image2.paste(poly,mask=poly)
    
    open_cv_image = np.array(image2) 
    #open_cv_image = cv2.putText(open_cv_image, text, (350,440), cv2.FONT_HERSHEY_SIMPLEX, 
    #               1, (153, 255, 51), 2, cv2.LINE_AA)
    open_cv_image = cv2.putText(open_cv_image, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (153, 255, 51), 2, cv2.LINE_AA)
    #open_cv_image = cv2.putText(open_cv_image, text2, (360,450), cv2.FONT_HERSHEY_SIMPLEX, 
    #               0.9, (204, 204, 0), 2, cv2.LINE_AA)
    resizedImage1 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    resizedImage2 = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)
    # numpy_horizontally = np.hstack((image, open_cv_image))
    numpy_horizontally = np.concatenate((resizedImage1, resizedImage2),axis = 0)
    # cv2.namedWindow("Output", cv2.WND_PROP_FUjkjkjLLSCREEN)
    # cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Output", numpy_horizontally)
    
    #cv2.imshow("Masked Image", masked_image)
    #cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        camera.release()
        break

camera.release()
cv2.destroyAllWindows()
print("Total Time:",total_time-time.time())











