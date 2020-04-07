import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from imageio import imread, imwrite
import os
import math
import random


parser = argparse.ArgumentParser(description='VFX_Hw1_HDR')
parser.add_argument('-i', '--original_img_path', default='./5/', type=str)# image path

args = parser.parse_args()


class HDR (object):
    def __init__(self,args):
        self.img_path = args.original_img_path
        self.image_list = []

#======================Read Images =========================

    def ReadImage(self,path):
        
        print(path)
        for filename in os.listdir(path):
            #print(filename)
            img= cv2.imread(os.path.join(path,filename),1)
            if img is not None:
                self.image_list.append(img)

        print('Number of Image =', len(self.image_list))
        print('image shape:', self.image_list[0].shape)

#====================Alignment===============================
# TODO
# 1.Threshold of alignment
# 2.save image after alignment
#
    def Alignment(self, depth = 6):
        
        AlignmentImage = []

        h, w = self.image_list[0].shape[:2]
        #self.image_list[1] = cv2.warpAffine(self.image_list[4], np.float32([[1, 0, 96],[0, 1,90]]), (w, h))
        
        gray_img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.image_list]
        #print(len(gray_img_list))

        #for i in range(len(gray_img_list)):
        #    cv2.imwrite(str(i)+'gray.jpg',gray_img_list[i])

        median = [np.median(img) for img in gray_img_list]
        
        binary_thres_img = [cv2.threshold(gray_img_list[i], median[i], 255, cv2.THRESH_BINARY)[1] for i in range(len(gray_img_list))]
        #print(gray_img_list[1])
        #h, w = gray_img_list[0].shape[:2]
        #gray_img_list[0] = cv2.warpAffine(gray_img_list[1], np.float32([[1, 0, 99],[0, 1,-20]]), (w, h))
        #print(gray_img_list[0])

        for i in range(0, len(binary_thres_img)):# every image need alignment
            #print(i)
            
            x, y = 0, 0 
            
            for d in range(depth,-1,-1): # 1/16 => 1/8 => 1/4 =>1/2
                x, y = x*2 ,y*2
                min_error = np.inf
                best_dx, best_dy = 0, 0
                #print(d)
                h, w = gray_img_list[0].shape[:2]
                
                standard_img = cv2.resize(binary_thres_img[4], (0, 0), fx=1/(2**d), fy=1/(2**d))
                now_img = cv2.resize(binary_thres_img[i], (0, 0), fx=1/(2**d), fy=1/(2**d))
                #cv2.imwrite(str(d)+'depth.jpg',standard_img)
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
            

            h, w = gray_img_list[0].shape[:2]
            AlignmentImage.append(cv2.warpAffine(self.image_list[i], np.float32([[1, 0, x],[0, 1, y]]), (w, h)))
            print(np.array(AlignmentImage).shape)

        
        #print(median,binary_thres_img[0])
        #plt.imshow(mask_img[0], cmap='gray')
        #plt.show()
        return AlignmentImage
#============================HDR=====================================
        
    def LoadImgHDR(self,path):
        filenames = []
        exposure_times = []
        f = open(os.path.join(path, 'exposure.txt'))
        for line in f:
            if (line[0] == '#'):
                continue
            #print(line)
            (filename, exposure, *rest) = line.split()
            filenames.append(filename)
            #print(filename)
            exposure_times.append(exposure)

        self.image_list = [cv2.imread(os.path.join(path, f), 1) for f in filenames]
        #把檔案讀進來後直接做alignment
        img_list_temp = self.Alignment(depth = 6)
        

        #=======================如果必要，做resize 以減少computation cost
        
        #img_list = []
        
        #for i in range(len(img_list_temp)):
        #    img_list.append(cv2.resize(img_list_temp[i],(1500,1000),interpolation = cv2.INTER_CUBIC))
        
        #=========================================


        img_list_b = [img[:,:,0] for img in img_list]
        img_list_g = [img[:,:,1] for img in img_list]
        img_list_r = [img[:,:,2] for img in img_list]
        exposure_times = np.array(exposure_times).astype(float)
        
        return (img_list_b, img_list_g, img_list_r, exposure_times)

    def Paul_Debvec_Method(self, img_list, exposure_times):
        # Z(i,j) is the pixel values of pixel location number i in image j 
        # B(j) is the log delta t, or log shutter speed, for image j 
        # l is lamdba, the constant that determines the amount of smoothness 
        # w(z) is the weighting function value for pixel value z 

        # Returns: 
        # g(z) is the log exposure corresponding to pixel value z 
        # lE(i) is the log film irradiance at pixel location i 
        
        
        sample_img = [cv2.resize(img, (10, 10)) for img in img_list]
        Z = [img.flatten() for img in sample_img]
        '''
        w_indexlist = range(0,999)
        h_indexlist = range(0,999)
        sample_w = random.sample(w_indexlist,200)
        sample_h = random.sample(h_indexlist,200)
        print(sample_h)
        print(sample_w)
        Z = []
        for i in range(len(img_list)):
            temp = []
            for j in range(100):
                temp.append(img_list[i][sample_w[j]][sample_h[j]])
            Z.append(temp)
        '''
        print(np.array(Z).shape)
        
        #===========
        B = [math.log(e,2) for e in exposure_times]
        l = 50
        # weight function define
        w = [z if z <= 0.5*255 else 255-z for z in range(256)]

        return self.gsolve(Z, B, l, w)

    def gsolve(self, Z, B, l, w):
        
        n = 256
        A = np.zeros(shape=(np.array(Z).shape[0]*np.array(Z).shape[1]+n+1, n+np.array(Z).shape[1]), dtype=np.float32)
        b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)


        # Include the data−fitting equations
        k = 0
        for i in range(np.array(Z).shape[1]):
            for j in range(np.array(Z).shape[0]):
                wij = w[Z[j][i]]
                A[k][Z[j][i]] = wij
                A[k][n+i] = -wij
                b[k] = wij*B[j]
                k += 1
        
        # Fix the curve by setting its middle value to 0
        A[k][128] = 1
        k += 1

        # Include the smoothness equations
        for i in range(n-1):
            A[k][i]   =    l*w[i+1]
            A[k][i+1] = -2*l*w[i+1]
            A[k][i+2] =    l*w[i+1]
            k += 1

        # Solve the system using SVD
        x = np.linalg.lstsq(A, b)[0]
        g = x[:256]
        lE = x[256:]

        return g, lE

#===================Reconstruction E================================

    def Radiance_Map(self,g, Z, ln_t, w):
        acc_E = [0]*len(Z[0])
        ln_E = [0]*len(Z[0])
        
        pixels, imgs = len(Z[0]), len(Z)
        for i in range(pixels):
            acc_w = 0
            for j in range(imgs):
                z = Z[j][i]
                acc_E[i] += w[z]*(g[z] - ln_t[j])
                acc_w += w[z]
            ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else acc_E[i]
            acc_w = 0
        
        return ln_E

    def Reconstruction_E(self,img_list, Response_Curve, exposure_times):
        img_size = img_list[0][0].shape
        hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')
        ln_t = np.log2(exposure_times)
        w = [z if z <= 0.5*255 else 255-z for z in range(256)]
        
        
        vfunc = np.vectorize(lambda x:math.exp(x))
    
        
        for i in range(3):
            print(' - Compute Radiance map: channel {0} .... \n'.format('BGR'[i]), end='', flush=True)
            Z = [img.flatten().tolist() for img in img_list[i]]
            E = self.Radiance_Map(Response_Curve[i], Z, ln_t, w)
            hdr[..., i] = np.reshape(vfunc(E), img_size)
            

        return hdr
 

    #https://gist.github.com/edouardp/3089602
    def HDR_Image_Save(self,hdr, filename):
        image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
        image[..., 0] = hdr[..., 2]
        image[..., 1] = hdr[..., 1]
        image[..., 2] = hdr[..., 0]

        f = open(filename, 'wb')
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1]) 
        f.write(bytes(header, encoding='utf-8'))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 256.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)
        f.close()

#=========================Tone mapping=================================
    #========global tone mapping=========
    def ToneMappingGlobal(self, radiance_map, d=1e-6, a=0.5):
        # d: log zero avoidance
        # a: average key 整體圖片的亮度
        # Lw:多少亮度以上變成255
        Lw = radiance_map
        Lw_average = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_average) * Lw
        Lm_white = np.max(Lm) # Lm_white, intensity that larger than this value will set to 1
        Ld = (Lm * (1 + (Lm / (Lm_white ** 2)))) / (1 + Lm)
        result = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)

        cv2.imwrite(self.img_path[2:]+"_"+"tone_mapping_global.jpg", result)
        return result

    #========local tone mapping=========
    # a: average key
    # fi: sharpening
    # epsilon: threshold of smoothness
    #s_max: s find to  what value
    def My_Gaussian(self,im, s_max=25, a=1.0, fi=8.0, epsilon=0.01):
        cols, rows = im.shape
        blur_prev = im
        num_s = int((s_max+1)/2)
        
        blur_list = np.zeros(im.shape + (num_s,))
        Vs_list = np.zeros(im.shape + (num_s,))
        
        for i, s in enumerate(range(1, s_max+1, 2)):
            
            blur = cv2.GaussianBlur(im, (s, s), 0)
            Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
            blur_list[:, :, i] = blur
            Vs_list[:, :, i] = Vs
        
        # 2D index
        s_max = np.argmax(Vs_list > epsilon, axis=2)
        s_max[np.where(s_max == 0)] = 1
        s_max -= 1
        
        # select blur size for each pixel
        I, J = np.ogrid[:cols, :rows]
        blur_s_max = blur_list[I, J, s_max]

        return blur_s_max

    #normal tone mappinf
    def ToneMappingLocal(self,radiance_map, d=1e-6, a=0.5):
        result = np.zeros_like(radiance_map, dtype=np.float32)
        weights = [0.065, 0.67, 0.265]
        
        
        Lw_ave = np.exp(np.mean(np.log(d + radiance_map)))
        
        for c in range(3):
            Lw = radiance_map[:, :, c]
            Lm = (a / Lw_ave) * Lw
            Ls = self.My_Gaussian(Lm)
            Ld = Lm / (1 + Ls)
            result[:, :, c] = np.clip(np.array(Ld * 255), 0, 255).astype(np.uint8)
        
        cv2.imwrite(self.img_path[2:]+"_"+"tone_mapping_local.jpg", result)
        return result


    def ToneMappingLocal_Radial(self,radiance_map, d=1e-6, a=0.5):
        result = np.zeros_like(radiance_map, dtype=np.float32)
        
        
        
        Lw = 0.065 * radiance_map[:, :, 0] + 0.67 * radiance_map[:, :, 1] + 0.265 * radiance_map[:, :, 2]
        Lw_ave = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_ave) * Lw
        Ls = self.My_Gaussian(Lm)
        Ld = Lm / (1 + Ls)
        
        for c in range(3):
            result[:, :, c] = np.clip(np.array((Ld / Lw) * radiance_map[:, :, c] * 255), 0, 255).astype(np.uint8)
        
        cv2.imwrite(self.img_path[2:]+"_"+"tone_mapping_local_radial.jpg", result)
        return result



def main(args):
    HDR_Pipeline = HDR(args)
    origin_img_path = args.original_img_path

    
    #=========Alignment=======================
    #HDR_Pipeline.ReadImage(origin_img_path)
    #AlignmentImage = HDR_Pipeline.Alignment(depth = 6)
    
    #=========HDR_Paul_Debvec_Method=============================
    print("\tRead Image For HDR")
   
    ImgListB, ImgListG, ImgListR, ExposureTimes = HDR_Pipeline.LoadImgHDR(origin_img_path)
    #print(ImgListB[0])
    #print(ImgListG[0])
    #print(ImgListR[0])
    #print(ExposureTimes)
    print("\tLoad Image Done\n\tCompute Responce Curve:")
    gFunctionB, _ = HDR_Pipeline.Paul_Debvec_Method(ImgListB, ExposureTimes)
    gFunctionG, _ = HDR_Pipeline.Paul_Debvec_Method(ImgListG, ExposureTimes)
    gFunctionR, _ = HDR_Pipeline.Paul_Debvec_Method(ImgListR, ExposureTimes)
    #print(gFunctionG)
    
    #===================HDR_Show_Responce_Curve=====================
    plt.figure(figsize=(10, 10))
    plt.plot(gFunctionR, range(256), 'rx')
    plt.plot(gFunctionG, range(256), 'gx')
    plt.plot(gFunctionB, range(256), 'bx')
        
    plt.savefig(origin_img_path[2:]+"_"+'ResponseCurve.png')
    
    print("\tCompute Responce Curve Done")

    #==================Reconstruction E=============================
    hdr = HDR_Pipeline.Reconstruction_E([ImgListB, ImgListG, ImgListR], [gFunctionB, gFunctionG, gFunctionR], ExposureTimes)
    
    

    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig(origin_img_path[2:]+"_"+'radiance-map.png')
    

    HDR_Pipeline.HDR_Image_Save(hdr,origin_img_path[2:]+"_"+"HDR.hdr")
    
    
    
    print("HDR Done")
    #======================tone mapping=============================
    res = HDR_Pipeline.ToneMappingGlobal(hdr)
    res = HDR_Pipeline.ToneMappingLocal(hdr)
    res = HDR_Pipeline.ToneMappingLocal_Radial(hdr)


if __name__ == '__main__':
    main(args) 