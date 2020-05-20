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

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

detector = MTCNN()

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
                    if rect[2] < 90 or rect[3] < 90:
                        size_face_small = True
                    #show result for checking
                    #if show_result:                               
                        #image_show = cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)

                    #if crop area around face
                    crop_image = True

                    a = rect[2] / 112
                    b = rect[3] / 112
                    w_crop = int(124*a)
                    h_crop = int(124*b)
                    w_12 = int((w_crop-rect[2]) / 2)
                    h_12 = int((h_crop - rect[3]) /2)

                    if crop_image:
                        roi = []
                        #roi[0]
                        if rect[0] >= w_12:
                            roi.append(rect[0] - w_12)
                        else:
                            roi.append(0)
                        #roi[1]
                        if rect[1] >= h_12:
                            roi.append(rect[1] - h_12)
                        else:
                            roi.append(0)
                        #roi[2]
                        if (W - (rect[0]+rect[2]) >= w_12):
                            roi.append(rect[2] + rect[0] + w_12 - roi[0])
                        else:
                            roi.append( W - roi[0])
                        #roi[3]
                        if (H - (rect[1]+rect[3]) >= h_12):
                            roi.append(rect[3] + rect[1] + h_12 - roi[1])
                        else:
                            roi.append( H - roi[1])

                        # if rect[0] >= 10:
                        #     roi.append(rect[0] - 10)
                        # else:
                        #     roi.append(0)
                        # #roi[1]
                        # if rect[1] >= 10:
                        #     roi.append(rect[1] - 10)
                        # else:
                        #     roi.append(0)
                        # #roi[2]
                        # if (W - (rect[0]+rect[2]) >= 10):
                        #     roi.append(rect[2] + 20)
                        # else:
                        #     roi.append( W - roi[0])
                        # #roi[3]
                        # if (H - (rect[1]+rect[3]) >= 10):
                        #     roi.append(rect[3]) + 20)
                        # else:
                        #     roi.append( H - roi[1])

                    else:
                        roi = rect
                    
                    image_roi = image[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
                    print('shape of image :',image_roi.shape)
                    if face_in_image and not size_face_small:
                        image_roi = cv2.resize(image_roi,(124,124),interpolation = cv2.INTER_AREA)

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

                    if not face_in_image:
                        num_image_not_include_face +=1
                        sub = os.path.join(not_face,folder)
                        if not os.path.isdir(sub):
                            os.mkdir(sub)
                        output_file = os.path.join(sub,file)
                        cv2.imwrite(output_file,image_roi)
                    
                    if show_result:
                        cv2.imshow('roi',image_roi)                             
                        image_show = cv2.rectangle(image,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),(255,0,0),2)
            
            #show result for checking
            if show_result:
                print(results)
                cv2.imshow('Display',image_show)
                cv2.waitKey(0)
            
print('number image include face ok : %d' % (num_image_include_face_ok))
print('number image not include face : %d' % (num_image_not_include_face))
print('number image small size : %d' % (num_image_small_size))



