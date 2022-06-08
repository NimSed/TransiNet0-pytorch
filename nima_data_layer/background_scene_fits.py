"""
Copyright © Nima Sedaghat 2017-2021

All rights reserved under the GPL license enclosed with the software. Over and
above the legal restrictions imposed by this license, if you use this software
for an academic publication then you are obliged to provide proper attribution
to the below paper:

    Sedaghat, Nima, and Ashish Mahabal. "Effective image differencing with
    convolutional neural networks for real-time transient hunting." Monthly
    Notices of the Royal Astronomical Society 476, no. 4 (2018): 5365-5376.
"""

from astropy.convolution import convolve,convolve_fft
from astropy.io import fits
from astropy import stats
import random
import numpy as np
from numpy.random import normal
from numpy.random import poisson
from skimage import io,color,transform,util
from skimage import img_as_ubyte,img_as_float
import warnings
import sys
import cv2

from psf import *
from general_tools import mMnorm,my_min_max_stats

eps = 1e-7;
fits_dimming_constant = 1.0#100000.0

class background_scene:


    def __init__(self,params):
        self.fits_mode = True
        self.params = params;
        self.extract_input_params()

        #-- Set caching up
        self.cache_enabled = self.params.get("cache_enabled",False)
        if self.cache_enabled:
            self.cache = {}

        data_path_prefix = self.params.get("data_path_prefix","")
        
        #-- Get list(s) of backgrounds and truth images 
        with open(self.params["back_images_list"],'rt') as f:
            self.back_images_list = f.read().splitlines();
        self.back_images_list = [data_path_prefix+x for x in self.back_images_list] #add prefix to file paths

        if self.params.get("back_images_list2",None) is not None:
            with open(self.params["back_images_list2"],'rt') as f:
                self.back_images_list2 = f.read().splitlines();
            self.back_images_list2 = [data_path_prefix+x for x in self.back_images_list2]  #add prefix to file paths
        else:
            self.back_images_list2 = None;

        if self.params.get("custom_diff_list",None) is not None:
            with open(self.params["custom_diff_list"],'rt') as f:
                self.custom_diff_list = f.read().splitlines();
            self.custom_diff_list = [data_path_prefix+x for x in self.custom_diff_list] #add prefix to file paths
        else:
            self.custom_diff_list = None;


        self.shuffle_lists = self.params.get("shuffle_lists",True)
    #-------------------------------------------------
    def extract_input_params(self):
        self.highres_container_size = [self.params["big_container_size"],self.params["big_container_size"]]
        self.final_img_size = [self.params["final_image_size"],self.params["final_image_size"]]
        self.maxGalaxyImages = self.params["maxGalaxyImages"]
        self.galaxy_crop_size = self.params.get("GalaxyCropSize",None)
        self.galaxy_scale_stats = self.params["GalaxyScaleStats"]
        self.galaxy_translation_stats = self.params["GalaxyTranslationStats"]
        self.galaxy_translation_distribution_mode = self.params.get("GalaxyTranslationDistributionMode","normal") # uniform/normal
        self.galaxy_rand_rotation = self.params.get("GalaxyRandRotation",False);
        self.galaxy_augmentation_method = self.params.get("GalaxyAugmentationMethod",'old');
        self.galaxy_transform_fill_blanks = self.params.get("galaxy_transform_fill_blanks",False);
        self.galaxy_skip_empty_diff_crops = self.params.get("galaxy_skip_empty_diff_crops",False);
        self.galaxy_skip_empty_diff_crops_max_attempts = self.params.get("galaxy_skip_empty_diff_crops_max_attempts",1000);
        self.galaxy_skip_padding = self.params.get("galaxy_skip_padding",False);
        self.MinNTransients = self.params["MinNTransients"]
        self.MaxNTransients = self.params["MaxNTransients"]
        self.MinCosmicRays = self.params.get("MinCosmicRays",0)
        self.MaxCosmicRays = self.params.get("MaxCosmicRays",0)
        self.CosmicRayAmpMean = self.params.get("CosmicRayAmpMean",1)
        self.CosmicRayAmpSigma = self.params.get("CosmicRayAmpSigma",0)        
        self.transient_location_stats = self.params["transient_location_stats"]
        self.transient_location_distribution_mode = self.params.get("transient_location_distribution_mode","normal")
        self.transient_amplitude_stats = np.array(self.params.get("transient_apmlitude_stats",(0,0)));
        self.transient_local_relative_magnitude_stats = self.params.get("transient_local_relative_magnitude_stats",(0,0))
        self.transient_onref_amplitude_stats = self.params.get("transient_onref_amplitude_stats",(0,0)) #if this param is set, then we may have transients on the reference image as well, which are at the exact same locations as the ones on the science image, to simulate variables or disappearing ones too
        self.right_transient_normalization = self.params.get("right_transient_normalization",False)
        self.bipolar_clipping = self.params.get("bipolar_clipping",False)
        self.no_clipping = self.params.get("no_clipping",False)        
        self.psf_sigma_stats = self.params["psf_sigma_stats"]
        self.psf_max_eccentricity = self.params["psf_max_eccentricity"];
        self.psf2_sigma_stats = self.params["psf2_sigma_stats"]; #In this version, psf2 params should be defined, not optional any more
        self.psf2_max_eccentricity = self.params["psf2_max_eccentricity"];
        self.psft_sigma_stats = self.params.get("psft_sigma_stats",(0,0));
        self.psft_max_eccentricity = self.params.get("psft_max_eccentricity",None);
        self.normalize_psf_for_apply = self.params.get("normalize_psf_for_apply",True); #this is true for backward compatibility
        self.median_master_stats = np.array(self.params["median_master_stats"])
        self.std_master_stats = np.array(self.params["std_master_stats"])
        self.median_query_stats = np.array(self.params["median_query_stats"])
        self.std_query_stats = np.array(self.params["std_query_stats"])
        self.noise_model = self.params.get("noise_model",'poisson')
        self.normalize_only_per_need = self.params.get("normalize_only_per_need",False);
        self.exact_copy_09 = self.params.get("exact_copy_09",False);
        self.KS = self.params.get("exact_copy_09_KS",0);
        self.use_fft_convolve = self.params.get("use_fft_convolve",False);
        self.MaxConstantSources = self.params.get("MaxConstantSources",0);
        self.MinConstantSources = self.params.get("MinConstantSources",0);
        self.ConstantSources_amplitude_stats = self.params.get("ConstantSources_apmlitude_stats",None)
        self.pairwise_aug_rotation_distribution_type = self.params.get("pairwise_aug_rotation_distribution_type",None); # uniform/normal
        self.pairwise_aug_rotation_distribution_params = self.params.get("pairwise_aug_rotation_distribution_params",(0,0)); # [a,b] for uniform distribution and [u,s] for normal
        self.pairwise_aug_translation_distribution_type = self.params.get("pairwise_aug_translation_distribution_type",None); # uniform/normal
        self.pairwise_aug_translation_distribution_params = self.params.get("pairwise_aug_translation_distribution_params",(0,0)); # [a,b] for uniform distribution and [u,s] for normal
        self.pairwise_aug_scale_distribution_type = self.params.get("pairwise_aug_scale_distribution_type",None); # uniform/normal
        self.pairwise_aug_scale_distribution_params = self.params.get("pairwise_aug_scale_distribution_params",(0,0)); # [a,b] for uniform distribution and [u,s] for normal
        self.pairwise_aug_fixed_x_translation = self.params.get("pairwise_aug_fixed_x_translation",None);#this is only used for graph generation
        self.outputVisuallySuitable = self.params.get("outputVisuallySuitable",False)
        self.VisScaleUsedImg = self.params.get("VisScaleUsedImg",2) #by default statistics of img2sn 
        self.VisScaleSigmaLow = self.params.get("VisScaleSigmaLow",3)
        self.VisScaleSigmaHigh = self.params.get("VisScaleSigmaHigh",10)
        self.syntheticOutputLocations = self.params.get("syntheticOutputLocations",False)        
        
        # A very important down-scaling of parameters
        # (these parameters should follow the temp down-scaling and up-scaling imposed on the image itself)
        if(self.fits_mode):
            self.transient_amplitude_stats = np.array(self.transient_amplitude_stats)/fits_dimming_constant
            self.median_master_stats /= fits_dimming_constant
            self.std_master_stats /= fits_dimming_constant
            self.median_query_stats /= fits_dimming_constant
            self.std_query_stats /= fits_dimming_constant

    #-------------------------------------------------
    def sample_transformation_params(self,ref_shape,distributions_params):
        ''' sample from the random transformation variables '''

        W = ref_shape[1];
        H = ref_shape[0];
        
        (rotation_distribution_type,rotation_distribution_params,\
         translation_distribution_type,translation_distribution_params,\
         scale_distribution_type,scale_distribution_params) = distributions_params;


        if(rotation_distribution_type == 'normal'):
            rotation = normal(loc=rotation_distribution_params[0],scale=rotation_distribution_params[1])
        elif(rotation_distribution_type == 'uniform'):
            rotation = random.uniform(rotation_distribution_params[0],rotation_distribution_params[1])
        else:
            rotation = 0;
            
        t1 = translation_distribution_params[0]
        t2 = translation_distribution_params[1]

        if(translation_distribution_type == 'normal'):
            translation = np.array([normal(loc=t1*W,scale=t2*W),normal(loc=t1*H,scale=t2*H)])
        elif(translation_distribution_type == 'uniform'):
            translation = np.array([random.uniform(t1*W,t2*W),random.uniform(t1*H,t2*H)])
        else:
            translation = np.array([0,0])


        if(scale_distribution_type == 'normal'):
            scale = normal(loc=scale_distribution_params[0],scale=scale_distribution_params[1])
        elif(scale_distribution_type == 'uniform'):
            scale = random.uniform(scale_distribution_params[0],scale_distribution_params[1])
        else:
            scale = 1;

        return scale,rotation,translation

    def sample_main_transformation_params(self,ref_size=None):

        if(ref_size is None):
            ref_size = self.highres_container_size #traditionally

        galaxy_rotation_distribution_type = 'uniform' if self.galaxy_rand_rotation else None #Traditionally we use unfiorm dist. for background rotation
        galaxy_rotation_distribution_params = [0,360]; #traditionally
        galaxy_translation_distribution_type = self.galaxy_translation_distribution_mode
        galaxy_translation_distribution_params = self.galaxy_translation_stats;
        galaxy_scale_distribution_type = 'normal' #traditionally
        galaxy_scale_distribution_params = self.galaxy_scale_stats;

        scale,rotation,translation = self.sample_transformation_params(ref_size,
                                                                  (galaxy_rotation_distribution_type,
                                                                   galaxy_rotation_distribution_params,
                                                                   galaxy_translation_distribution_type,
                                                                   galaxy_translation_distribution_params,
                                                                   galaxy_scale_distribution_type,
                                                                   galaxy_scale_distribution_params)
        )

        return scale,rotation,translation
       
    def sample_pairwise_aug_transformation_params(self):
        pair_aug_rotation_distribution_type = self.pairwise_aug_rotation_distribution_type
        pair_aug_rotation_distribution_params = self.pairwise_aug_rotation_distribution_params
        pair_aug_translation_distribution_type = self.pairwise_aug_translation_distribution_type
        pair_aug_translation_distribution_params = self.pairwise_aug_translation_distribution_params
        pair_aug_scale_distribution_type = self.pairwise_aug_scale_distribution_type
        pair_aug_scale_distribution_params = self.pairwise_aug_scale_distribution_params

        scale,rotation,translation = self.sample_transformation_params(self.highres_container_size,
                                                                  (pair_aug_rotation_distribution_type,
                                                                   pair_aug_rotation_distribution_params,
                                                                   pair_aug_translation_distribution_type,
                                                                   pair_aug_translation_distribution_params,
                                                                   pair_aug_scale_distribution_type,
                                                                   pair_aug_scale_distribution_params)
        )
        
        return scale,rotation,translation
            
            
    #-------------------------------------------------
    def transform_galaxy_image(self,
                               galaxy,output_shape,
                               scale,rotation,translation,
                               skip_padding,transform_fill_blanks):


        #- Resize the galaxy to the output shape
        if(galaxy.shape != output_shape):
            galaxy = transform.resize(galaxy,output_shape = output_shape, mode='constant', preserve_range=True);

        #- Scale/Zoom
        if(scale != 1):
            galaxy = transform.rescale(galaxy,scale,mode='constant', preserve_range=True)

        #- pad/crop based on the size
        if any(np.asarray(galaxy.shape) < output_shape):
            if(not transform_fill_blanks):
                galaxy = np.pad(galaxy, np.int16(np.ceil( (np.asarray(output_shape)-np.asarray(galaxy.shape))/2 ))
                                ,mode='constant',constant_values = 0)
            else:
                galaxy = np.pad(galaxy, np.int16(np.ceil( (np.asarray(output_shape)-np.asarray(galaxy.shape))/2 ))
                                ,mode='symmetric')

        if any(np.asarray(galaxy.shape) > output_shape): #we do cropping after padding, to account for inaccurate padding artifacts too
            d = np.asarray(galaxy.shape) - output_shape;
            galaxy = util.crop(galaxy,((d[0]//2,d[0]-d[0]//2),(d[1]//2,d[1]-d[1]//2)));


        #- Rotation -- should be done separately, as the SimilarityTransform does not supprt rotation around center
        if(rotation != 0):
            galaxy = transform.rotate(galaxy,rotation,mode='constant' if not transform_fill_blanks else 'symmetric',
            preserve_range=True);

        #- Translation
        shift = translation * -1;
        transform_matrix = transform.SimilarityTransform(translation=shift);
        galaxy = transform.warp(galaxy,transform_matrix,
                                mode='constant' if not transform_fill_blanks else 'symmetric',
                                cval=0, output_shape=galaxy.shape, preserve_range = True);
       
        return galaxy


    #-------------------------------------------------
    def load_image_from_cache(self,filename):
        try:
            return self.cache[filename]
        except: #if filename was not found in cache
            return None

    #-------------------
    def add_image_to_cache(self,filename,img):
        # we do not check if it is already in the cache.
        # the check should be performed already
        self.cache[filename] = img

    #-------------------
    def load_image_file(self,filename):

        # If cache_enabled and the file is already loaded into cache,
        # no need to reload it from disk
        if self.cache_enabled:
            img = self.load_image_from_cache(filename)
            if img is not None:
                return img

        #---- Load the file from disk
        if(self.fits_mode):
            hdu = fits.open(filename)[0]
            img = hdu.data

            #--- Take care of NaNs
            if(np.any(np.isnan(img))):
                SATURATE = hdu.header['SATURATE'] 
                img[np.isnan(img)] = SATURATE

            #--- Downscale to be able to use skimage functions (which require float in [-1,1])
            img = img/fits_dimming_constant #

        else:
            img = img_as_float(io.imread(filename,as_grey=True));


        #--- caching
        if self.cache_enabled:
            self.add_image_to_cache(filename,img)

        return img


    #-------------------------------------------------
    def crop_galaxy_center(self,img,crop_size,shift=[0,0]):
        y,x = img.shape
        x1 = x//2-(crop_size//2)-int(shift[0])
        y1 = y//2-(crop_size//2)-int(shift[1])
        x2 = x1+crop_size
        y2 = y1+crop_size
        out = img[y1:y2,x1:x2]
        assert(out.shape==(crop_size,crop_size))
        return out,(x1,y1,x2,y2)
        

    #-------------------------------------------------
    def add_galaxies(self,nS,index=None):

        for i in range(nS):

            #- Load the galaxy image
            if index is None:
                galaxy_file_index = random.randint(0,len(self.back_images_list)-1); #important note: if you use the numpy.random.randint, you should change the 2nd argument
            else:
                galaxy_file_index = index % len(self.back_images_list);
                
            galaxy_filename = self.back_images_list[galaxy_file_index];

            galaxy = self.load_image_file(galaxy_filename)

            #- Sample a transformation from the random space for the main background augmentation
            ''' Note: currently cropping is implemented very naively, in which the whole image is
            first transformed/augmented, and then the crop is taken from the center.
            Later this can/should be optimized '''

            if(self.galaxy_crop_size is None):
                sz = self.highres_container_size
            else:
                sz = galaxy.shape #resizing should be kept for after cropping, in case we do cropping

                
            #'''in a loop try to find a crop which includes a transient in it'''
            #(only if we have all three images as input, and the user has asked for it)
            if self.galaxy_skip_empty_diff_crops and self.custom_diff_list is not None:
                diff_filename = self.custom_diff_list[galaxy_file_index];
                custom_diff = self.load_image_file(diff_filename)

                for iteration in range(self.galaxy_skip_empty_diff_crops_max_attempts):
                    main_scale,main_rotation,main_translation = self.sample_main_transformation_params(ref_size=self.highres_container_size)

                    if custom_diff.max() < 1e-5: #Use the current *empty* crop already, since this is an empty-diff sample
                        dmx = custom_diff.max()
                        #print(diff_filename,' was empty!!')
                        break

                    temp_diff,coords = self.crop_galaxy_center(custom_diff,self.galaxy_crop_size,main_translation)

                    dmx = temp_diff.max()
                    #print(iteration,coords,' dddddmx:',dmx)
                    if dmx>1e-5:
                        # from matplotlib import pyplot as plt
                        # plt.imshow(temp_diff/temp_diff.max())
                        # plt.show(block=True)
                        break

            else: #Einfach find a transformation
                main_scale,main_rotation,main_translation = self.sample_main_transformation_params(ref_size=self.highres_container_size)


            #--- And crop out from the ref image
            if self.galaxy_augmentation_method == 'new': #New augmentation method, in which the crop box is transformed and then applied

                from direct_warp import warp_crop_direct
                galaxy = warp_crop_direct(galaxy,self.galaxy_crop_size,self.galaxy_crop_size,
                                          main_scale,main_rotation,main_translation,from_center=True,
                                          symmetric=self.galaxy_transform_fill_blanks)

            else: # In the old method each galaxy image is transformed and then cropped at the center

                # #- Transform the main galaxy image (and crop if needed)
                if self.galaxy_crop_size  and (galaxy.shape != sz or main_scale != 1 or main_rotation != 0): #if it is only translation, we can leave it for cropping
                    galaxy = self.transform_galaxy_image(galaxy,sz,
                                                         main_scale,main_rotation,main_translation,
                                                         self.galaxy_skip_padding,self.galaxy_transform_fill_blanks)

                if(self.galaxy_crop_size):
                    galaxy,coorddd = self.crop_galaxy_center(galaxy,self.galaxy_crop_size,shift=main_translation)
                    if(galaxy.shape != tuple(self.highres_container_size)):
                        galaxy = transform.resize(galaxy,output_shape = self.highres_container_size,mode='constant'); #because we skipped it a few lines above


            #- and add it to the container img
            self.img += galaxy;

            #--- Consider the secondary image too
            if(self.back_images_list2 is not None):
                galaxy_filename2 = self.back_images_list2[galaxy_file_index];
                galaxy2 = self.load_image_file(galaxy_filename2)
                if self.galaxy_augmentation_method == 'new': #New augmentation method, in which the crop box is transformed and then applied
                    galaxy2 = warp_crop_direct(galaxy2,self.galaxy_crop_size,self.galaxy_crop_size,
                                               main_scale,main_rotation,main_translation,from_center=True)

                else: # In the old method each galaxy image is transformed and then cropped at the center
                    #- Transform it
                    galaxy2 = self.transform_galaxy_image(galaxy2,sz,
                                                          main_scale,main_rotation,main_translation,
                                                          self.galaxy_skip_padding,self.galaxy_transform_fill_blanks)

                    #- and crop if needed
                    if(self.galaxy_crop_size):
                        galaxy2,_ = self.crop_galaxy_center(galaxy2,self.galaxy_crop_size)
                        if(galaxy2.shape != tuple(self.highres_container_size)):
                            galaxy2 = transform.resize(galaxy2,output_shape = self.highres_container_size,mode='constant'); #because we skipped it a few lines above

                self.img_secondary += galaxy2;

                
            #--- Consider custom diff (ground-truth) too
            if(self.params.get('custom_diff_list',None) is not None):
                diff_filename = self.custom_diff_list[galaxy_file_index];
                custom_diff = self.load_image_file(diff_filename)

                if self.galaxy_augmentation_method == 'new': #New augmentation method, in which the crop box is transformed and then applied
                    custom_diff = warp_crop_direct(custom_diff,self.galaxy_crop_size,self.galaxy_crop_size,
                                               main_scale,main_rotation,main_translation,from_center=True)

                else: # In the old method each galaxy image is transformed and then cropped at the center
                    #- Transform it
                    custom_diff = self.transform_galaxy_image(custom_diff,sz,
                                                              main_scale,main_rotation,main_translation,
                                                              self.galaxy_skip_padding,self.galaxy_transform_fill_blanks)

                    #- and crop if needed
                    if(self.galaxy_crop_size):
                        custom_diff,_ = self.crop_galaxy_center(custom_diff,self.galaxy_crop_size)
                        if(custom_diff.shape != tuple(self.highres_container_size)):
                            custom_diff = transform.resize(custom_diff,output_shape = self.highres_container_size,mode='constant'); #because we skipped it a few lines above


                self.total_custom_diff += custom_diff;

            
        self.img /= nS;

        if self.img_secondary is not None:
            self.img_secondary /= nS;
        if self.total_custom_diff is not None:
            self.total_custom_diff /= nS;

    #-------------------------------------------------
    def AddConstantStars(self):

        if(self.MaxConstantSources > 0): 
            x = [];
            y = [];
            nConstantSources = random.randint(self.MinConstantSources,self.MaxConstantSources)
            for k in range(nConstantSources):
                for tries in range(1000): #we want to make sure that the point sources are distant enough
                    x_ = random.randint(0,W);
                    y_ = random.randint(0,H)

                    if(k==0):
                        break;
                    else:
                        mindist = min([math.sqrt( (xx-x_)**2 + (yy-y_)**2 ) for xx,yy in zip(x,y)])
                        if mindist > W/4:
                            break;

                x_ = x_ if x_ < W else W-1
                y_ = y_ if y_ < H else H-1

                x.append(x_)
                y.append(y_)

                cu = self.ConstantSources_amplitude_stats[0];
                css = self.ConstantSources_amplitude_stats[1];
                c = self.normal(loc=cu,scale=css)

                self.img[y_,x_] += c;

                if self.img_secondary is not None:
                    self.img_secondary[y_,x_] += c;

    #-------------------------------------------------
    def AddTransients(self,target_img,transients_on_1,transients_on_2):

        def GetTransientPeakValue(amp_stats,local_mag_stats,back_image,x_,y_,psf_to_apply,fwhm_to_have,sky_to_add):
            if local_mag_stats[0] == 0 and local_mag_stats[1] == 0: #This is the traditional case where we simply set the pixel-value
                au = amp_stats[0];
                ass = amp_stats[1];
                a = normal(loc=au,scale=ass)
                a = 0 if a < 0 else a
                return a
            
            else: #This is the new case where we set the pixel-value based on the magnitude relative to the neighborhood
                if(amp_stats[0] != 0 or amp_stats[1] != 0):
                    sys.exit('You can not mix relative mag and traditional absolute amp! (2)');

                from flux import estimate_neighbourhood_flux
                bg_estimation_radius = fwhm_to_have*1.2/2;
                f_local = estimate_neighbourhood_flux(back_image,[x_,y_],bg_estimation_radius,psf_to_apply,self.conv);
                f_local += sky_to_add;

                #-- sample the relative magnitude from the provided distribution params
                mag_u = local_mag_stats[0];
                mag_s = local_mag_stats[1];
                mag_relative = normal(loc=mag_u,scale=mag_s)

                #-- compute the simulated transient's flux based on f_local and mag_relative
                f_trans = f_local * (2.512**(-mag_relative)) - f_local;

                #-- for now 
                a = f_trans*(fwhm_to_have**2 * 4);
                    
                #print('mag: ',f_local*255,f_trans*255,(f_trans+f_local)*255,a*255)
                return a;
            
        
        # let's use a single psf for all the transients in the scene
        psft = gaussian_psf(self.psft_sigma_stats,self.psft_max_eccentricity);

        tt1 = self.transient_location_stats[0];
        tt2 = self.transient_location_stats[1];
    
        x = [];
        y = [];

        nT = random.randint(self.MinNTransients,self.MaxNTransients) if self.MaxNTransients >= self.MinNTransients else 0;
        W = self.highres_container_size[1]; 
        H = self.highres_container_size[0];

        for k in range(nT):

            for tries in range(1000): #we want to make sure that the point sources are distant enough
                if(self.transient_location_distribution_mode == 'normal'):
                    x_ = round(W/2 + normal(loc=tt1*W,scale=tt2*W));
                    y_ = round(H/2 + normal(loc=tt1*H,scale=tt2*H));
                elif(self.transient_location_distribution_mode == 'uniform'):
                    x_ = round(W/2 + random.uniform(tt1*W,tt2*W));
                    y_ = round(H/2 + random.uniform(tt1*H,tt2*H));
                else:
                    translation = np.array([0,0])

                if(k==0):
                    break;
                else:
                    mindist = min([math.sqrt( (xx-x_)**2 + (yy-y_)**2 ) for xx,yy in zip(x,y)])
                    if mindist > W/2:
                        break;

            x_ = x_ if x_ < W else W-1
            y_ = y_ if y_ < H else H-1

            x.append(x_)
            y.append(y_)

            # sample the amplitude of the transient on the second image
            a2 = GetTransientPeakValue(self.transient_amplitude_stats,
                                       self.transient_local_relative_magnitude_stats,
                                       self.img1,x_,y_,
                                       psf_to_apply=self.psf2,
                                       fwhm_to_have= math.sqrt(self.psf2.get_max_fwhm()**2+psft.get_max_fwhm()**2),
                                       sky_to_add=self.sampled_noise_params[2]/255 if self.noise_model == 'correct_real_poisson' else self.sampled_noise_params[2]);

            # and if necessary also on the first (ref) image
            if self.transient_onref_amplitude_stats != (0,0):
                if(self.transient_local_relative_magnitude_stats is not None):
                    sys.exit('You can not mix relative mag and traditional absolute amp! (1)');
                else:
                    a1 = GetTransientPeakValue(self.transient_onref_amplitude_stats,
                                               self.transient_local_relative_magnitude_stats,
                                               self.img1,x_,y_,
                                               psf_to_apply=self.psf1,
                                               fwhm_to_have= math.sqrt(self.psf1.get_max_fwhm()**2+psft.get_max_fwhm()**2),
                                               sky_to_add=self.sampled_noise_params[0]/255 if self.noise_model == 'correct_real_poisson' else self.sampled_noise_params[0]);
            else:
                a1 = 0;


            psft.add_to_image(transients_on_1,x_,y_,a1)
            psft.add_to_image(transients_on_2,x_,y_,a2)
            if not self.syntheticOutputLocations:
                psft.add_to_image(target_img,x_,y_,a2-a1)
            else: #generate a gaussian to represent the location -- should follow the exact specs used when generating loc-based gt from reals
                psfl = gaussian_psf([2,2],0);
                psfl.add_to_image(target_img,x_,y_,mag=1)



        return target_img,transients_on_1,transients_on_2

    #-------------------------------------------------
    def AddCosmicRays(self,target_img):
        
        psfc = gaussian_psf([0,0],0); #This is to simulate a delta function

    
        x = [];
        y = [];

        nT = random.randint(self.MinCosmicRays,self.MaxCosmicRays) if self.MaxCosmicRays >= self.MinCosmicRays else 0;
        W = self.highres_container_size[1]; 
        H = self.highres_container_size[0];

        for k in range(nT):

            for tries in range(1000): #we want to make sure that the point sources are distant enough
                x_ = round(random.uniform(0,W-1));
                y_ = round(random.uniform(0,H-1));

                if(k==0):
                    break;
                else:
                    mindist = min([math.sqrt( (xx-x_)**2 + (yy-y_)**2 ) for xx,yy in zip(x,y)])
                    if mindist > W/2:
                        break;

            x_ = x_ if x_ < W else W-1
            y_ = y_ if y_ < H else H-1

            x.append(x_)
            y.append(y_)

            # Obtain a random amp for the cosmic ray
            au = self.CosmicRayAmpMean
            ass = self.CosmicRayAmpSigma
            a = normal(loc=au,scale=ass)
            a = 0 if a < 0 else a

            psfc.add_to_image(target_img,x_,y_,a)


        return target_img

    #-------------------------------------------------
    def SampleNoiseParams(self):
        u1 = normal(self.median_master_stats[0],self.median_master_stats[1])
        s1 = normal(self.std_master_stats[0],self.std_master_stats[1])
        u2 = normal(self.median_query_stats[0],self.median_query_stats[1])
        s2 = normal(self.std_query_stats[0],self.std_query_stats[1])

        u1 = 0 if u1 < 0 else u1
        s1 = 1e-7 if s1 < 0 else s1 
        u2 = 0 if u2 < 0 else u2
        s2 = 1e-7 if s2 < 0 else s2

        return u1,s1,u2,s2
        
    def AddNoise(self,NoiseParams=None):

        if NoiseParams is None:
            u1,s1,u2,s2 = SampleNoiseParams()
        else:
            u1,s1,u2,s2 = NoiseParams
            
        if self.noise_model == 'gaussian':
            self.img1sn = self.img1s + normal(u1,s1,self.img1s.shape)
            self.img2sn = self.img2s + normal(u2,s2,self.img2s.shape)
        elif self.noise_model == 'correct_real_poisson':
            assert(self.fits_mode)
            self.img1sn = poisson(self.img1s*1.0+u1)/1.0
            self.img2sn = poisson(self.img2s*1.0+u2)/1.0
        else:
            sys.exit('wrong noise model in data layer');

    #-------------------------------------------------
    def ClipConvertTo8Bit(self):

        self.img1sn = np.clip(self.img1sn,0,1);
        self.img2sn = np.clip(self.img2sn,0,1);
        self.dimgs = np.clip(self.dimgs,0,1);
        self.dimgs_normal = np.clip(self.dimgs_normal,0,1);

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.img1s = img_as_ubyte(self.img1s)
            self.img2s = img_as_ubyte(self.img2s)
            self.img1sn = img_as_ubyte(self.img1sn)
            self.img2sn = img_as_ubyte(self.img2sn)
            self.dimgs = img_as_ubyte(self.dimgs)
            self.dimgs_normal = img_as_ubyte(self.dimgs_normal);

    def biPolarClipScaleTo255(self):

        self.img1sn = np.clip(self.img1sn,0,1);
        self.img2sn = np.clip(self.img2sn,0,1);
        self.dimgs = np.clip(self.dimgs,-1,1); #only the clipping of dimg should differ in the biPolar clipping
        self.dimgs_normal = np.clip(self.dimgs_normal,-1,1);

        self.img1s *= 255
        self.img2s *= 255
        self.img1sn *= 255
        self.img2sn *= 255
        self.dimgs *= 255
        self.dimgs_normal *= 255

    def rescaleFitsData(self):
        self.img1s *= fits_dimming_constant
        self.img2s *= fits_dimming_constant
        self.img1sn *= fits_dimming_constant
        self.img2sn *= fits_dimming_constant
        self.dimgs *= fits_dimming_constant
        self.dimgs_normal *= fits_dimming_constant

    def JustScaleTo256(self):
        self.img1s *= 255
        self.img2s *= 255
        self.img1sn *= 255
        self.img2sn *= 255
        self.dimgs *= 255
        self.dimgs_normal *= 255

    def ScaleToVisual(self):

        m,M,_,_ = my_min_max_stats(self.img2sn if self.VisScaleUsedImg == 2 else self.img1sn,
                                   self.VisScaleSigmaLow,self.VisScaleSigmaHigh)

        self.img1s = mMnorm(self.img1s,m,M)
        self.img2s = mMnorm(self.img2s,m,M)
        self.img1sn = mMnorm(self.img1sn,m,M)
        self.img2sn = mMnorm(self.img2sn,m,M)
        self.dimgs = mMnorm(self.dimgs,m,M,it_is_diff=True)
        self.dimgs_normal = mMnorm(self.dimgs_normal,m,M,it_is_diff=True)

        
    #-------------------------------------------------
    def generate(self,args):

        #unfold the input arguments
        # note that if we are not in the parallel mode, instead of seed in fact a reliable counter is passed
        seed,q = (args[0],args[1]);        

        #--- reset the seeds (in parallel runs, it is crucial to do this based on index)
        from datetime import datetime
        from numpy.random import seed as npseed
        random.seed(str(datetime.now())+str(seed))
        npseed(int(datetime.now().timestamp())+(seed));

        #-- get a pointer to the right convolve function
        self.conv = convolve if not self.use_fft_convolve else convolve_fft

        #------------- Add the galaxy images
        nS = random.randint(1,self.maxGalaxyImages);
        self.img = np.zeros(self.highres_container_size);
        self.img_secondary = np.zeros_like(self.img) if self.back_images_list2 is not None else None
        self.total_custom_diff = np.zeros_like(self.img) if self.params.get('custom_diff_list',None) is not None else None

        if not self.params.get("real_parallel",False) and not self.params.get("shuffle_lists",True):
            self.add_galaxies(nS,seed) #in the non-parallel mode we can use a counter, if user does not want to do shuffling
        else:
            self.add_galaxies(nS)
            
        #------------- Add the constant stars
        self.AddConstantStars()

        #------------ Create the reference (master) image --new: don't apply the psf yet
        self.img1 = np.copy(self.img);

        #------------ Create the diff image
        self.dimg = np.zeros_like(self.img)
        self.synth_dimg = np.zeros_like(self.img)
        
        #-- add transient star(s) (to the synthetic diff image)
        # add the necessary components to the two images too
        #
        # We also need to know what sky levels we will be adding,
        # for the sake of transient_local_relative_magnitude to work,
        # so we sample noise params here
        transients_on_1 = np.zeros_like(self.synth_dimg);
        transients_on_2 = np.zeros_like(self.synth_dimg);

        self.sampled_noise_params = self.SampleNoiseParams();
        self.psf1 = gaussian_psf(self.psf_sigma_stats,self.psf_max_eccentricity);
        self.psf2 = gaussian_psf(self.psf2_sigma_stats,self.psf2_max_eccentricity);

        self.synth_dimg,transients_on_1,transients_on_2 = self.AddTransients(self.synth_dimg,transients_on_1,
                                                                             transients_on_2)
        self.dimg += self.synth_dimg

        #--- custom diff image: it is *NOT ADDED TO IMG2*, but just to the diff image
        if(self.total_custom_diff is not None):
            self.dimg += self.total_custom_diff

        #------------ Create the second image
        if self.img_secondary is None:
            self.img2 = np.copy(self.img);
        else:
            self.img2 = np.copy(self.img_secondary);

        #-- *ADD ONLY THE SYNTHETIC DIFF* image to img2 (new: only the component that's necessary)
        self.img2 += transients_on_2

        #-- new version: add the first component to img1
        self.img1 += transients_on_1
           
        #----- Apply the PSFs
        self.img1s = self.psf1.apply_on(self.img1,self.conv,normalize_kernel=self.normalize_psf_for_apply)
        self.img2s = self.psf2.apply_on(self.img2,self.conv,normalize_kernel=self.normalize_psf_for_apply)
        if not self.syntheticOutputLocations: #if it is a localization gt, it shall not go through psfs, etc.
            self.dimgs = self.psf2.apply_on(self.dimg,self.conv,normalize_kernel=self.normalize_psf_for_apply)
        else:
            self.dimgs = self.dimg

        #print('--> sumd:',self.dimgs.sum())

        #- also create and return a peak-normalized version of the ground truth
        if(self.right_transient_normalization):
            self.dimgs_normal = np.copy(self.dimgs)
            self.dimgs_normal[self.dimgs_normal < 0.01] = 0
            self.dimgs_normal /= (self.dimgs_normal+eps);
            self.dimgs_normal *= (self.dimgs);
        else:
            self.dimgs_normal = self.dimgs / (np.max(self.dimgs)+eps);  


        #--- Apply the pairwise augmentation -- only augment ref image
        pairaug_scale,pairaug_rotation,pairaug_translation = self.sample_pairwise_aug_transformation_params();
        if(self.pairwise_aug_fixed_x_translation is not None): #only for graph generation
            pairaug_translation[0] = self.pairwise_aug_fixed_x_translation;

        self.img1s = self.transform_galaxy_image(self.img1s,self.img1s.shape,
                                            pairaug_scale,pairaug_rotation,pairaug_translation,
                                            self.galaxy_skip_padding,self.galaxy_transform_fill_blanks)

        #--- Add cosmic rays
        self.img2s = self.AddCosmicRays(self.img2s);
        
        #--- add noise
        self.AddNoise(NoiseParams=self.sampled_noise_params)

        #--- Normalize and convert to 8bit greyscale images
        if self.fits_mode:
            if(self.outputVisuallySuitable):
                self.ScaleToVisual();
            else:
                self.rescaleFitsData();
        else:
            if self.no_clipping:
                self.JustScaleTo256();
            elif not self.bipolar_clipping:
                self.ClipConvertTo8Bit()
            else:
                self.biPolarClipScaleTo255()

        if q is not None:
            q.put( (self.img1sn,self.img2sn,self.dimgs,self.img1s,self.img2s,self.dimgs_normal) )
        else:
            return (self.img1sn,self.img2sn,self.dimgs,self.img1s,self.img2s,self.dimgs_normal)

