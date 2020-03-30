import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from imageio import imread, imwrite
import os

parser = argparse.ArgumentParser(description='VFX_Hw1_HDR')
parser.add_argument('-i', '--img_path', default='./data/test/', type=str)# image path

args = parser.parse_args()


class HDR (object):
    def __init__(self,args):
        self.img_path = args.img_path
        self.image_list = []

#======================Read Images =========================

    def ReadImage(self):
        
        
        print(self.img_path)
        for filename in os.listdir(self.img_path):
            #print(filename)
            img= cv2.imread(os.path.join(self.img_path,filename))
            if img is not None:
                self.image_list.append(img)

        print('Number of Image =', len(self.image_list))
        print('image shape:', self.image_list[0].shape)

#====================Alignment===============================


    def Alignment(self, depth = 6):
        gray_img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.image_list]
        #print(len(gray_img_list))

        for i in range(len(gray_img_list)):
            cv2.imwrite(str(i)+'gray.jpg',gray_img_list[i])

        median = [np.median(img) for img in gray_img_list]
        
        binary_thres_img = [cv2.threshold(gray_img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(gray_img_list))]
        #print(gray_img_list[1])
        #h, w = gray_img_list[0].shape[:2]
        #gray_img_list[0] = cv2.warpAffine(gray_img_list[1], np.float32([[1, 0, 99],[0, 1,-20]]), (w, h))
        print(gray_img_list[0])

        for i in range(0, len(binary_thres_img)):# every image need alignment
            #print(i)
            
            x, y = 0, 0 
            
            for d in range(depth,-1,-1): # 1/16 => 1/8 => 1/4 =>1/2
                x, y = x*2 ,y*2
                min_error = np.inf
                best_dx, best_dy = 0, 0
                #print(d)
                h, w = gray_img_list[0].shape[:2]
                
                standard_img = cv2.resize(binary_thres_img[0], (0, 0), fx=1/(2**d), fy=1/(2**d))
                now_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1/(2**d), fy=1/(2**d))
                #print("---",2**d)
                ignore_pixels = np.ones(standard_img.shape)
                ignore_pixels[np.where(np.abs(standard_img - median[i]) <= 4)] = 0

                h, w = standard_img.shape[:2]
                #print("===============")
                #print(standard_img)
                #print(now_img)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        now_img_affine = cv2.warpAffine(now_img, np.float32([[1, 0, x + dx],[0, 1, y+dy]]), (w, h))
                        error = np.abs(np.sign(standard_img - now_img_affine))
                        #error = np.sum(error * ignore_pixels)
                        error = np.sum(error)
                        
                        if error < min_error:
                            min_error = error
                            #print(error, " ", dx, " ", dy," d:",d)
                            best_dx, best_dy = dx, dy

                #print(x," ",y," ",best_dx," ",best_dy)
                x, y = x+best_dx, y+best_dy

            print("Image",i,": best_dx: ", x,"best_dy ",y)

        #print(median,binary_thres_img[0])
        #plt.imshow(mask_img[0], cmap='gray')
        #plt.show()

        


def main(args):
    HDR_Pipeline = HDR(args)

    HDR_Pipeline.ReadImage()
    HDR_Pipeline.Alignment(depth = 6)
    


if __name__ == '__main__':
    main(args)