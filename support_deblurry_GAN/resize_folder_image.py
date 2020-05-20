import os
import sys
import cv2
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

for folder in sub_directorys:
    sub_folder = os.path.join(input_folder,folder)
    if not os.path.isdir(sub_folder):
        continue
    sub_folder_out = os.path.join(output_folder,folder)
    if not os.path.isdir(sub_folder_out):
        os.mkdir(sub_folder_out)

    list_file = os.listdir(sub_folder)

    for file in list_file:
        if file.endswith(('.png','.jpg','jpeg','JPEG')):
            input_file = os.path.join(sub_folder, file)
            image = cv2.imread(input_file)
            print(input_file)

            image = cv2.resize(image,(100,100),interpolation = cv2.INTER_AREA)
            output_file = os.path.join(sub_folder_out,file)
            cv2.imwrite(output_file,image)




