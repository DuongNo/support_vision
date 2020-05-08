import os
import sys
import cv2
from mtcnn import MTCNN
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-in','--input_folder', action='store', dest='input_folder', default='./input/',
                    help='Path to input folder contain image')

parser.add_argument('-out','--output_folder', action='store', dest='output_folder', default='./output/',
                    help='Path to output folder save image')

values = parser.parse_args()

input_folder = values.input_folder
output_folder = values.output_folder

if not os.path.isdir(input_folder):
    print('input folder image not exist')
    sys.exit()
else:
    sub_directorys = os.listdir(input_folder)
     #print('number sub folder :',sub_directorys)
     #sys.exit()

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

detector = MTCNN()

#folder_input = '/home/duong/Documents/researching/GAN/common/image_enhance/input_image'
#folder_output = '/home/duong/Documents/researching/GAN/common/image_enhance/output'

face_ok = os.path.join(output_folder,'face_ok')
not_face = os.path.join(output_folder,'not_face')
small_face = os.path.join(output_folder,'small_face')

if not os.path.isdir(face_ok):
    os.mkdir(face_ok)
if not os.path.isdir(not_face):
    os.mkdir(not_face)
if not os.path.isdir(small_face):
    os.mkdir(small_face)

num_image_include_face_ok = 0
num_image_not_include_face = 0
num_image_small_size = 0
show_result = False

for folder in sub_directorys:
    sub_folder = os.path.join(input_folder,folder)
    if not os.path.isdir(sub_folder):
        continue
    list_file = os.listdir(sub_folder)

    for file in list_file:
        if file.endswith(('.png','.jpg','jpeg','JPEG')):
            input_file = os.path.join(sub_folder, file)

            image = cv2.imread(input_file)
            H = image.shape[0]
            W = image.shape[1]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img)

            face_in_image = False
            size_face_small = False
            for result in results:
                confiden = result['confidence']
                if confiden > 0.9:
                    face_in_image = True
                    rect = result['box']
                    if rect[2] < 60 or rect[3] < 60:
                        size_face_small = True
                    #show result for checking
                    #if show_result:                               
                        #image_show = cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)

                    #crop area around face
                    roi = []
                    #roi[0]
                    if rect[0] >= int(0.2 * rect[2]):
                        roi.append(rect[0] - int(0.2*rect[2]))
                    else:
                        roi.append(0)
                    #roi[1]
                    if rect[1] >= int(0.15 * rect[3]):
                        roi.append(rect[1] - int(0.15*rect[3]))
                    else:
                        roi.append(0)
                    #roi[2]
                    if (W - (rect[0]+rect[2]) >= int(0.2 * rect[2])):
                        roi.append(int(1.2*rect[2]) + rect[0] - roi[0])
                    else:
                        roi.append( W - roi[0])
                    #roi[3]
                    if (H - (rect[1]+rect[3]) >= int(0.15 * rect[3])):
                        roi.append(int(1.15*rect[3]) + rect[1] - roi[1])
                    else:
                        roi.append( H - roi[1])
                    
                    image_roi = image[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]

                    if face_in_image and not size_face_small:
                        num_image_include_face_ok +=1
                        sub = os.path.join(face_ok,folder)
                        if not os.path.isdir(sub):
                            os.mkdir(sub)
                        output_file = os.path.join(sub,file)
                        cv2.imwrite(output_file,image_roi)

                    if size_face_small:
                        sub = os.path.join(small_face,folder)
                        if not os.path.isdir(sub):
                            os.mkdir(sub)
                        output_file = os.path.join(sub, file)
                        cv2.imwrite(output_file,image_roi)
                        num_image_small_size +=1
                    
                    if show_result:
                        cv2.imshow('roi',image_roi)                             
                        image_show = cv2.rectangle(image,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),(255,0,0),2)
            
            #show result for checking
            if show_result:
                print(results)
                cv2.imshow('Display',image_show)
                cv2.waitKey(0)
            
            if not face_in_image:
                num_image_not_include_face +=1
                sub = os.path.join(not_face,folder)
                if not os.path.isdir(sub):
                    os.mkdir(sub)
                output_file = os.path.join(sub,file)
                cv2.imwrite(output_file,image_roi)




print('number image include face ok : %d' % (num_image_include_face_ok))
print('number image not include face : %d' % (num_image_not_include_face))
print('number image small size : %d' % (num_image_small_size))



