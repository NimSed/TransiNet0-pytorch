import numpy as np
from astropy.table import Table
import math
import random
from photutils.datasets import make_gaussian_sources_image


class psf_base:

    def __init__(self, kernel=np.atleast_2d([1])):
        self.kernel = kernel;

    def shape(self):
        return np.asarray(self.kernel.shape)

    def apply_on(self,img,conv,clip_values=None,normalize_kernel=True):
        if not all(self.shape() == (1,1)):
            imgS = conv(img,self.kernel,normalize_kernel=normalize_kernel);
        else:
            imgS = img * self.kernel

        #-- Clip pixel values, if required
        if clip_values is not None:
            imgS = np.clip(imgS,clip_values[0],clip_values[1])
            
        return imgS

    def add_to_image(self,img,x,y,mag):
        """ This method adds the PSF as a gaussian 2D shape to the image, and has nothing to do
        with correlation, etc. """
        
        if(any(img.shape < self.shape())):
            raise('PSF bigger than image!');
        
        k = np.copy(self.kernel);

        (h,w) = k.shape
        (H,W) = img.shape

        w2_ = w // 2;
        h2_ = h // 2;

        x0 = x - w2_;
        y0 = y - h2_;

        x1 = x0+w-1;
        y1 = y0+h-1;

        if x0 < 0:
            if x0+w-1 < 0:
                return img
            else:
                k = k[:,-x0:]
                x0=0;
        elif x1 > W-1:
            if x0 > W-1:
                return img
            else:
                k = k[:,0:W-x0]
        else:
            k = k;
            
        if y0 < 0:
            if y0+h-1 < 0:
                return img
            else:
                k = k[-y0:,:]
                y0 = 0;
        elif y1 > H-1:
            if y0 > H-1:
                return img
            else:
                k = k[0:H-y0,:]
        else:
            k = k;
            

        (h,w) = k.shape
        x1 = x0+w-1;
        y1 = y0+h-1;

        #--- finally add the kernel
        img[y0:y0+h,x0:x0+w] += mag*k;
        
        return img
        
#----------------------------------------------------
class gaussian_psf(psf_base):
    
    def __init__(self,psf_sigma_stats,psf_max_eccentricity):

        super(gaussian_psf,self).__init__(); #initialize a simple identity psf
        
        (psf_min_sigma,psf_max_sigma) = (psf_sigma_stats[0],psf_sigma_stats[1])

        # take care of a very special case in which the kernel is left as an indentity one
        if psf_min_sigma == psf_max_sigma == 0:
            self.sigma_x = 0;
            self.sigma_y = 0;
            self.theta = 0;
            return

        psf_matrix_dim = math.ceil(psf_max_sigma*7./2.)*2+1 #we need an odd kernel size!
        psf_shape = (psf_matrix_dim,psf_matrix_dim);

        
        table = Table()
        table['amplitude'] = [1]
        table['x_mean'] = [psf_matrix_dim//2]
        table['y_mean'] = [psf_matrix_dim//2]


        rx = random.uniform(psf_min_sigma,psf_max_sigma);
        if(psf_max_eccentricity is not None):
            e = random.uniform(0,psf_max_eccentricity);
            ry = rx * math.sqrt(1-e**2)
        else:
            ry = random.uniform(psf_min_sigma,psf_max_sigma);

        self.sigma_x = rx;
        self.sigma_y = ry;
        self.theta = [random.uniform(0,2*math.pi)]

        table['x_stddev'] = self.sigma_x
        table['y_stddev'] = self.sigma_y
        table['theta'] = self.theta
        
        self.kernel = make_gaussian_sources_image(psf_shape, table)
        return

    def get_max_fwhm(self):
        return max(self.sigma_x,self.sigma_y)*2.355

    
#----------------------------------------------------
class oldstyle_gaussian_psf(gaussian_psf):
    
    def __init__(self,KS,s11,s12,s22):

        super(gaussian_psf,self).__init__(); #initialize a simple identity psf

        def generate_gaussian(mu,sigma,imgSize):
            from scipy.stats import multivariate_normal
            y,x = np.mgrid[1:imgSize[0]+1,1:imgSize[1]+1]
            rv = multivariate_normal(mean=mu,cov=sigma)
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = y;
            pos[:, :, 1] = x
            p = rv.pdf(pos)
            p = p / np.max(p[:])
            return p

        uKernel = [KS/2,KS/2];    
        sKernel = np.array([[s11,s12],[s12,s22]]).T
        self.kernel = generate_gaussian(uKernel,sKernel,[KS,KS])

        #If the kernel dimensions are not odd, we shoudl pad with zeros to be compatible with MATLAB
        if psf.shape[0] % 2 == 0:
            self.kernel = np.pad(psf,[[1,0],[0,0]],'constant',constant_values=0);            
        if psf.shape[1] % 2 == 0:
            self.kernel = np.pad(psf,[[0,0],[1,0]],'constant',constant_values=0);            

        return
