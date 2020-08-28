import numpy as np
from scipy import ndimage

def harris_corner(img, k=0.04):
    '''
    img: grayscale image
    k: a score equation parameter [lower values detects more corners]
    '''
    Ix, Iy = np.gradient(img)

    W = np.array([[1,2,1],
                  [2,4,2],
                  [1,2,1]])/16      # Gaussian Window
    
    Sx2 = ndimage.convolve(Ix**2, W)
    Sy2 = ndimage.convolve(Iy**2, W)
    Sxy = ndimage.convolve(Ix*Iy, W)

    dst = Sx2*Sy2 - np.square(Sxy) - k*np.square(Sx2 + Sy2)
    return dst

if __name__ == "__main__":
    img = np.array([[1,0,0,0,0],
                    [0,1,0,0,0],
                    [0,0,1,1,1],
                    [1,1,1,0,0],
                    [1,1,0,0,0]])
    
    dst = harris_corner(img)
    print(dst)