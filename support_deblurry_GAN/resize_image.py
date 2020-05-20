import cv2

input_path = '/home/duong/Pictures/TC6.jpg'
output_path = '/home/duong/Pictures/TC6-x32.jpg'
img_size = 32
img = cv2.imread(input_path)
img =cv2.resize(img,(img_size,img_size))
cv2.imwrite(output_path,img)
