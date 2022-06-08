import numpy as np
from astropy import stats

def subtract_sky(data):
        from astropy.stats import sigma_clipped_stats
        from photutils import make_source_mask

        mk = make_source_mask(data, snr=2, npixels=5, dilate_size=11)
        mean,median,std = sigma_clipped_stats(data, sigma=3.0, mask=mk)
        data = data - median;

        return data

#-----------------------------------------------------
def cut_out_rect(data,xc,yc,w_2,h_2,do_padding=False):

    if do_padding:
        data = np.pad(data,( (int(h_2),int(h_2)),(int(w_2),int(w_2)) ),'reflect')
        xc = xc+w_2
        yc = yc+h_2

    x1 = int(round(max(0,xc-w_2)))
    y1 = int(round(max(0,yc-h_2)))
    x2 = int(round(min(data.shape[1],xc+w_2)))
    y2 = int(round(min(data.shape[0],yc+h_2)))
            
    cut = data[y1:y2,x1:x2].copy();

    xc_shift = int(round(xc-w_2-x1))
    yc_shift = int(round(yc-h_2-y1))

    #- If padding has occured, only for the sake of the returned rect,
    #  we need to rollback the coordinates.
    if do_padding:
        x1 -= w_2
        y1 -= h_2
        x2 -= w_2
        y2 -= h_2
    
    return cut,x1,y1,x2,y2,(xc_shift,yc_shift);

#-----------------------------------------------------
def mMnorm(img,m,M,it_is_diff=False):
    if not it_is_diff:
        img = np.clip(img,m,M)
        img -= m

    if m != M:
        img = img/(M-m)*255
    else:
        pass #All the pixels would already be set to 0
        
    img = np.clip(img,0,255)
    return img

#-----------------------------------------------------
def my_min_max_stats(img,sigma_low,sigma_high):

    #u = np.median(img)
    #s = np.std(img)
    _, u, s = stats.sigma_clipped_stats(img,sigma_lower=sigma_low,sigma_upper=sigma_high,iters=1)

    m = u-sigma_low*s
    M = u+sigma_high*s

    return m,M,u,s

#-----------------------------------------------------
def scalePairToVisual(img0,img1,reference_item,cut_below,cut_above):

    m,M,_,_ = my_min_max_stats(img0 if reference_item == 0 else img1,cut_below,cut_above)
    vimg0 = mMnorm(img0,m,M)
    vimg1 = mMnorm(img1,m,M)
    
    return vimg0,vimg1

