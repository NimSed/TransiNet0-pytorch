import numpy as np
from math import ceil as ceil
from scipy import signal
from astropy.convolution import convolve, convolve_fft

from numpy.ctypeslib import ndpointer
import ctypes
import os

downsample_lib = ctypes.CDLL(os.path.expanduser('~')+'/transinet-pytorch/pytorch/downsample_ctypes/downsample.so')
downsample_lib.DownsampleFeatures.argtypes = [ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                              ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

def triangular_2d_kernel(w,h): #by Nima
    kernel = np.zeros((h,w))

    wwindow = signal.triang(w)
    hwindow = signal.triang(h)

    for y in range(h):
        kernel[y,:w] = wwindow
        
    for x in range(w):
        kernel[:h,x] *= hwindow


    return kernel

def DownsampleFeaturesConv( nthreads,  num,  channels,  bottomwidth,  bottomheight,
                            topheight,  topwidth,  bot_countpernum,  bot_numstride,  widthScale,
                            heightScale,  wradius,  hradius, src_data, dest_data):

    dest_data = np.zeros([num,channels,topheight,topwidth],dtype='float32')

    kernel = triangular_2d_kernel(hradius*2+1,wradius*2+1)
    for n in range(num):
        for c in range(channels):
            blurred = convolve_fft(src_data[n,c],kernel)
            dest_data[n,c] = blurred[0::round(heightScale),0::round(widthScale)]

        

    return dest_data
            
def DownsampleFeatures( nthreads,  num,  channels,  bottomwidth,  bottomheight,
                        topheight,  topwidth,  bot_countpernum,  bot_numstride,  widthScale,
                        heightScale,  wradius,  hradius, src_data, dest_data):

    dest_data = np.zeros([num,channels,topheight,topwidth],dtype='float32')

    for n in range(num):
        for c in range(channels):
            for desty in range(topheight):
                for destx in range(topwidth):

                    #Compute source center pos in topdiff
                    botx = (np.float32(destx)/np.float32(topwidth-1)) * np.float32(bottomwidth-1)
                    boty = (np.float32(desty)/np.float32(topheight-1)) * np.float32(bottomheight-1)
    
                    ibotx = round(botx);
                    iboty = round(boty);
    
                    # Accumulate in range around that point:
                    botidxoffcn = (bot_numstride*n) + (bottomwidth*bottomheight*c);
    
                    accum_value = 0;
                    accum_weight = 0;

                    for yoff in range(-hradius,hradius+1) :
                        by = iboty + yoff;
                        botidxoffycn = by*bottomwidth + botidxoffcn;

                        for xoff in range(-wradius,wradius+1):
                            bx = ibotx + xoff;
                
                            if(bx >= 0 and by >= 0 and bx < bottomwidth and by < bottomheight):
                                #sample = src_data[bx + botidxoffycn];
                                sample = src_data[n,c,by,bx]
                                weight = max(0,1-(abs(np.float32(bx) - botx)/widthScale)) * max(0,1- (abs(np.float32(by) - boty)/heightScale) );

                                accum_value += sample * weight;
                                
                                if(destx == 1 and desty == 0 and sample != 0 and weight != 0):
                                    print("bx/M:%f\n"%(np.float32(bx)/widthScale *1000000));
                                    print("by/M:%f\n"%(np.float32(by)/heightScale *1000000));
                                    print("botxM:%f\n"%((botx) *1000000));
                                    print("botyM:%f\n"%((boty) *1000000));
                                    print("/wsM:%f\n"%( (abs(np.float32(bx) - botx)/widthScale) *1000000));
                                    print("/hsM:%f\n"%( (abs(np.float32(by) - boty)/heightScale) *1000000));
                                    print("wM:%f\n"%(weight*1000000));
                                    print("%f*%f->%f\n"%(sample,weight,accum_value))
                                    
                                accum_weight += weight;
                                
                    dest_data[n,c,desty,destx] = accum_value / accum_weight;
    
    return dest_data

def downsample_layer(bottom,top,conv=False,clib=False):

    topwidth = top.shape[3]
    topheight = top.shape[2]
    topchannels = top.shape[1]
    topcount = top.shape[0]

    bottomnum = 1
    bottomwidth = bottom.shape[3]
    bottomheight = bottom.shape[2]
    bottomchannels = bottom.shape[1]
    bottomcount = bottom.shape[0]

    if (bottomwidth != topwidth or bottomheight != topheight):

        #From bottom to top
        bot_countpernum = bottomwidth * bottomheight * bottomchannels
        bot_numstride = bottomwidth * bottomheight * bottomchannels
    
        widthScale = np.float32(bottomwidth-1) / np.float32(topwidth-1)
        heightScale = np.float32(bottomheight-1) / np.float32(topheight-1)
    
        wradius = ceil(widthScale)
        hradius = ceil(heightScale)

        if clib:
            downsample_lib.DownsampleFeatures(topcount,bottomnum,
                                              bottomchannels, bottomwidth, bottomheight, 
                                              topheight, topwidth, bot_countpernum, bot_numstride,
                                              widthScale, heightScale, wradius, hradius,
                                              np.ascontiguousarray(bottom), np.ascontiguousarray(top));
            return top
        
        else:
            if not conv:
                return DownsampleFeatures(topcount,
                                          bottomnum, bottomchannels, bottomwidth, bottomheight, 
                                          topheight, topwidth, bot_countpernum, bot_numstride, widthScale, heightScale, wradius, hradius, bottom, top);
            else:
                return DownsampleFeaturesConv(topcount,
                                              bottomnum, bottomchannels, bottomwidth, bottomheight, 
                                              topheight, topwidth, bot_countpernum, bot_numstride, widthScale, heightScale, wradius, hradius, bottom, top);

    else:
        return bottom



def downsample_layer_by_shape(bottom,out_shape,conv=False,clib=False):

    assert not(conv and clib)
    
    fake_top = np.zeros(out_shape,dtype='float32')
    
    if(bottom.ndim < 4):
        bottom = np.expand_dims(bottom,0)
    if(bottom.ndim < 4):
        bottom = np.expand_dims(bottom,0)

    return downsample_layer(bottom,fake_top,conv,clib)
