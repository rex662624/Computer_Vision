import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from imageio import imread, imwrite

parser = argparse.ArgumentParser(description='ACV hw1')
parser.add_argument('-ia', '--img_a', default='./trucka.bmp', type=str)
parser.add_argument('-ib', '--img_b', default='./truckb.bmp', type=str)
parser.add_argument('-io', '--img_output', default='./output.jpg', type=str)
parser.add_argument('-bs', '--block_size', default=8, type=int)
parser.add_argument('-sr', '--search_range', default=50, type=int)
args = parser.parse_args()

class Motion(object):
    def __init__(self,args):
        self.img_a = imread(args.img_a).astype(np.int)
        self.img_b = imread(args.img_b).astype(np.int)
        self.block_size = args.block_size
        self.search_range = args.search_range
        

    def Compute(self):
        print(self.block_size," ",self.search_range," ",self.block_size)
        print(self.img_a.shape)
        
        Number_of_Height = self.img_a.shape[0] // self.block_size#高要移動多少次
        Number_of_Width = self.img_a.shape[1] // self.block_size#寬要移動多少次

        #store the value of the final xy displacement. shape represents for:block_w of index, block_h of index, xy coordinate    
        displacement = np.zeros(shape=(Number_of_Width, Number_of_Height, 2), dtype=np.int)


        for h_block_index in range(Number_of_Height):
            h_start = h_block_index * self.block_size
            h_end = h_start + self.block_size
            for w_block_index in range(Number_of_Width):
                w_start = w_block_index * self.block_size
                w_end = w_start + self.block_size
                
                small_block_a = self.img_a[w_start:w_end,h_start:h_end]
                min_diff = np.inf
                min_coordinate = None
                #上面已經切好a圖上的小window,下面去b圖一個一個search 找出最相近的small block

                for dh in range(max(-self.search_range,-h_start),min(self.search_range, self.img_b.shape[1] - 1 - h_end) + 1):#找 +- search range 內的pixel
                    for dw in range(max(-self.search_range,-w_start),min(self.search_range, self.img_b.shape[0] - 1 - w_end) + 1):#找 +- search range 內的pixel
                        small_block_b = self.img_b[w_start + dw : w_end + dw , h_start + dh : h_end + dh]
                        #print(dw," ",dh ,w_start," " ,w_end," ",h_start," " ,h_end)
                        diff = np.sum(np.abs(small_block_a-small_block_b))
                        #print(diff)
                        if (diff < min_diff):
                            min_diff = diff
                            min_coordinate = [dw,dh]
                #print(min_coordinate)
                displacement [w_block_index,h_block_index, : ] = min_coordinate
        
        H, W = np.meshgrid(np.arange(0, Number_of_Height), np.arange(0, Number_of_Width))
        ax = plt.figure(figsize=(10, 10)).gca()
        ax.set_title('Block Size: {} x {}'.format(self.block_size, self.block_size))
        ax.quiver(H, W, displacement[:, :, 1], displacement[:, :, 0], angles='xy')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(args.img_output)

def main(args):
    Motion(args).Compute()

if __name__ == '__main__':
    main(args)



