//gcc -shared -Wl,-soname,downsample -o downsample.so -fPIC downsample.c

#include <stdio.h>

#define Dtype float

#include <stdlib.h>
#include <math.h>
#define max(a,b) fmax(a,b)

float __int_as_float(int a)
{
    union {int a; float b;} u;
    u.a = a;
    return u.b;
}


void DownsampleFeatures(const int nthreads, const int num,
			const int channels, const int bottomwidth, const int bottomheight,
			const int topheight, const int topwidth, const int bot_countpernum, const int bot_numstride,
			const float widthScale, const float heightScale, const int wradius, const int hradius,
			const Dtype* src_data, Dtype* dest_data);


void DownsampleFeatures(const int nthreads, const int num,
			const int channels, const int bottomwidth, const int bottomheight,
			const int topheight, const int topwidth, const int bot_countpernum, const int bot_numstride,
			const float widthScale, const float heightScale, const int wradius, const int hradius,
			const Dtype* src_data, Dtype* dest_data)
{
    long N = num*channels*topwidth*topheight;

    for(long index = 0 ; index < N; index++)
    {
	int destx = index % topwidth; //w-pos
	int desty = (index / topwidth) % topheight; //h-pos
	
	int cn = (index / topwidth / topheight);
	int c = cn % channels; //channel
	int n = cn / channels; //num

	//Compute source center pos in topdiff
	float botx = ((float)destx/(float)(topwidth-1)) * (float)(bottomwidth-1); // \in [0.0, (topwidth-1)]
	float boty = ((float)desty/(float)(topheight-1)) * (float)(bottomheight-1);
    
	int ibotx = round(botx);
	int iboty = round(boty);

	// Accumulate in range around that point:
	int botidxoffcn = (bot_numstride*n) + (bottomwidth*bottomheight*c);
    
	float accum_value = 0;
	float accum_weight = 0;
	float accum_nan = 0;

	for(int yoff = -hradius; yoff <= hradius; yoff++)
	{
	    int by = iboty + yoff;
	    int botidxoffycn = by*bottomwidth + botidxoffcn;
	    for(int xoff = -wradius; xoff <= wradius; xoff++)
	    {
		int bx = ibotx + xoff;
			    
		if(bx >= 0 && by >= 0 && bx < bottomwidth && by < bottomheight)
		{
		    float sample = src_data[bx + botidxoffycn];
		    float weight = max(0.0f,1.0f-(abs((float)bx - botx)/widthScale)) * max(0.0f,1.0f- (abs((float)by - boty)/heightScale) );
		    if(sample != sample)//isnan
		    { 
			accum_nan += weight;
			sample = 0;
			weight = 0;
		    }
				
		    accum_value += sample * weight;
		    accum_weight += weight;

		}
	    }
	}
	if(accum_nan / accum_weight > 0.5)
	    dest_data[index] = __int_as_float(0x7fffffff); //CUDART_NAN_F;
	else
	    dest_data[index] = accum_value / accum_weight;
		    
    }
}
