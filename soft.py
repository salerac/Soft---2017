
# coding: utf-8

# In[1]:


from __future__ import print_function
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import collections
import keras
import math
import random
import datetime
from PIL import Image, ImageOps
from copy import deepcopy

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from numpy import loadtxt
from keras.models import load_model
import matplotlib.pylab as pylab
from tensorflow.python.client import device_lib

from scipy import ndimage


# In[2]:


def load_img(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 50, 255, cv2.THRESH_BINARY)#127
    return image_bin
def invert(image):
    return 255-image
def padd_region(region,w,h,i):
    cv2.imwrite("resized%d.jpg" %i, region)
    image = Image.fromarray(region)
    com = ndimage.measurements.center_of_mass(region)
    x = int(round(com[0]))
    y = int(round(com[1]))
    return np.array(ImageOps.expand(image, border=(14-y,14-x,14-(w-y),14-(h-x)))) #(14-x,14-y,14-x,14-y)


# In[3]:


def matrix_to_vector(image):
    return image.flatten()


# In[4]:


def scale_to_range(image):
    return image/255


# In[5]:


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    i = 0
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) 
        area = cv2.contourArea(contour)
        if area > 19 and h < 100 and h > 2 and w > 2:
            region = image_bin[y:y+h, x:x+w]
            padded = padd_region(region,w,h,i)
            i += 1
            cv2.imwrite("region.jpg", padded)
            regions_array.append([padded, (x,y,w,h)])
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    sorted_coords = [region[1] for region in regions_array]
    return image_orig, sorted_regions,sorted_coords


# In[6]:


def prepare_for_ann(regions):
    ready_for_ann = []
    i = 0
    for region in regions:
        cv2.imwrite("region%d.jpg"%i, region)
        i += 1
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


# In[7]:


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(10)
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


# In[8]:


def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


# In[9]:


def play_video(i,numbers,tracked_numbers,ann,alphabet,tracked_changes,copy,file):
    video = cv2.VideoCapture('videos/video-%d.avi'%i)
    success,image = video.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)    
        success,image = video.read()
        count += 1
    for j in range(1199):
        frame = load_img("frame%d.jpg" %j)
        numbers, tracked_numbers, tracked_changes, copy, ret = convert_frame(frame, numbers, tracked_numbers,ann,alphabet, tracked_changes, j, copy)
        for number in ret:
            crossed.append(number)
    print(datetime.datetime.now())
    print("BROJEVI KOJI SU PRESLI LINIJU: ",crossed)
    print("SUMA VIDEO-%d: "%i,sum(crossed))
    suma = sum(crossed)
    file.write("\nvideo-%d.avi\t%d"%(i,suma))


# In[10]:


def convert_frame(frame, numbers, tracked, ann, alphabet, tracked_changes, frame_number, copy):
    crossed = []
    (x1,y1,x2,y2) = find_line(frame)
    points = find_line_points(x1,y1,x2,y2)
    img = image_bin(image_gray(frame))
    #print(display_image(img))
    selected_regions, numbers, coords = select_roi(img.copy(), img)
    if frame_number == 1:
        copy.clear()
        for t in tracked:
            copy.append(deepcopy(t))
    #for tracked_number in tracked:
        #print(tracked_number.number, tracked_number.center)     
    if frame_number%30 == 0 and frame_number >= 10:
        indexes_to_pop = []
        for j in range(len(copy)):
            if copy[j].center[0] == tracked[j].center[0] and copy[j].center[1] == tracked[j].center[1]:
                #print("Broj %d"%copy[j].number," ce biti uklonjen")
                tracked_changes[j] += 1
                indexes_to_pop.append(j)
        count = 0        
        for index in indexes_to_pop:
            tracked.pop(index-count)
            count += 1
        copy.clear()        
        for t in tracked:
            copy.append(deepcopy(t))       
        
        
    for i in range(len(numbers)):
        numbers1 = []
        numbers1.append(numbers[i])
        flag = 0
        (x,y,w,h) = coords[i]
        x_center = x + round(w/2)
        y_center = y + round(h/2)
        center = (x_center,y_center)
        #print(center)
        for tracked_number in tracked:
            if tracked_number.check_distance(x_center,y_center):
                for (x,y) in points:
                    if x-9 <= tracked_number.center[0] <=x+9 and y-10 <= tracked_number.center[1] <= y+10:
                        if tracked_number.presao == False:
                            tracked_number.set_presao()
                            crossed.append(tracked_number.number)
                tracked_number.change_center(center)
                flag = 1
        if flag == 0:
            if check_nearest(center, tracked):
                if check_close(points,x_center,y_center, frame_number):
                    number = Number()
                    number.id = random.randint(1,101)
                    number.change_center(center)
                    number.image = numbers[i]
                    number.set_number(display_result(ann.predict(np.array(prepare_for_ann(numbers1))),alphabet))
                    #print("dodat")
                    tracked.append(number)
                    tracked_changes.append(0)
    numbers = prepare_for_ann(numbers1)
    return numbers, tracked, tracked_changes, copy, crossed


# In[11]:


def check_close(points,x_center,y_center, frame):
    for(x,y) in points:
        if x_center > x-2 and y_center > y-2: #30 all #x-9 <= x_center <=x+9 and y-11 <= y_center <= y+11
            return False
    return True


# In[12]:


def find_line_points(x1,y1,x2,y2):
    k = (y2 - y1)/(x2 - x1)
    b = y1 - k*x1
    ret = []
    for y in range(y2,y1,1):
        x = round((y-b)/k)
        point = (x,y)
        ret.append(point)
    return ret


# In[13]:


def check_nearest(center, numbers):
    if len(numbers) == 0:
        return True
    for tracked_number in numbers:
        distance = math.sqrt( ((center[0]-tracked_number.center[0])**2)+((center[1]-tracked_number.center[1])**2) )
        if distance < 35: #40 #26 #35
            return False
    return True


# In[14]:


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


# In[15]:


def find_line(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 300
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 6)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('houghlines5.jpg',frame)
    return (x1,y1,x2,y2)


# In[16]:


class Number:
    id = None
    center = []
    image = []
    number = None;
    predicted = False
    presao = False
    
    def change_center(self, new_center):
        self.center = new_center
    def check_distance(self,x,y):
        distance = math.sqrt( ((x-self.center[0])**2)+((y-self.center[1])**2) )
        if round(distance) < 20:
            return True
        else:
            return False
    def set_predicted(self):
        self.predicted = True
        
    def set_number(self,number):
        self.number = number[0]
        #print(self.number, " dodat na poziciji ",self.center[0],self.center[1], " ID %d" %self.id)
        self.set_predicted()
    def set_presao(self):
        self.presao = True
        #print("Broj %d" %self.number, " presao liniju na poziciji ",self.center[0],self.center[1], " ID %d" %self.id)
        


# In[17]:


alphabet = [0,1,2,3,4,5,6,7,8,9]
ann = load_model("ann1.h5")


# In[18]:


tracked_numbers = []
tracked_changes = []
copy = []
crossed = []
numbers = []
file = open(r"out.txt","w")
file.write("RA 122/2013 Aleksandar Rac\n")
file.write("file\tsum")


# In[19]:


for i in range(10):
    play_video(i,numbers,tracked_numbers,ann,alphabet,tracked_changes,copy,file)
    tracked_numbers = []
    tracked_changes = []
    copy = []
    crossed = []
    numbers = []
file.close()
exec(compile(open("test.py", "rb").read(), "test.py", 'exec'))

