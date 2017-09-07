/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */
#include "stdafx.h"
#include <cmath>
#include <iostream>
#include <cv.h>

using namespace cv;
using namespace std;

#include <time.h>
#include <stdio.h>

double getTime()
{     
#ifdef _POSIX_CPUTIME                                                                                                                                         
   struct timespec ts;
   if (!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts))
   {
      return (double)(ts.tv_sec) + (double)(ts.tv_nsec)/1.0e9;
   } else
#endif                                                                                                                                                 
   {
      // fall back to standard unix time
      return 0;
   }
}

template <typename ValueType>
void swap(ValueType *a, ValueType *b)
{
   ValueType tmp = *a; *a = *b; *b = tmp;
}

void solveLinear3x3(float *A, float *b)
{
   // find pivot of first column
   int i = 0;
   float *pr = A;
   float vp = abs(A[0]); 
   float tmp = abs(A[3]);   
   if (tmp > vp) 
   { 
      // pivot is in 1st row
      pr = A+3; 
      i = 1; 
      vp = tmp; 
   }
   if (abs(A[6]) > vp) 
   { 
      // pivot is in 2nd row
      pr = A+6;
      i = 2;
   }
   
   // swap pivot row with first row
   if (pr != A) { swap(pr, A); swap(pr+1, A+1); swap(pr+2, A+2); swap(b+i, b); }
   
   // fixup elements 3,4,5,b[1]
   vp = A[3] / A[0]; A[4] -= vp*A[1]; A[5] -= vp*A[2]; b[1] -= vp*b[0];
   
   // fixup elements 6,7,8,b[2]]
   vp = A[6] / A[0]; A[7] -= vp*A[1]; A[8] -= vp*A[2]; b[2] -= vp*b[0];
   
   // find pivot in second column
   if (abs(A[4]) < abs(A[7])) { swap(A+7, A+4); swap(A+8, A+5); swap(b+2, b+1); }
   
   // fixup elements 7,8,b[2]
   vp = A[7] / A[4];
   A[8] -= vp*A[5];
   b[2] -= vp*b[1];

   // solve b by back-substitution
   b[2] = (b[2]                    )/A[8];
   b[1] = (b[1]-A[5]*b[2]          )/A[4];
   b[0] = (b[0]-A[2]*b[2]-A[1]*b[1])/A[0];
}

void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22)
{
   double a = a11, b = a12, c = a21, d = a22;   
   double det = sqrt(abs(a*d-b*c));
   double b2a2 = sqrt(b*b + a*a);
   a11 = b2a2/det;             a12 = 0;
   a21 = (d*b+c*a)/(b2a2*det); a22 = det/b2a2;   
}

void rectifyAffineTransformationUpIsUp(float *U)
{
   rectifyAffineTransformationUpIsUp(U[0], U[1], U[2], U[3]);
}

void computeGaussMask(Mat &mask)
{
   int size = mask.cols;
   int halfSize = size >> 1;
   // fit 3*sigma into half_size
   float scale = float(halfSize)/3.0f;
   
   float scale2 = -2.0f * scale * scale;   
   float *tmp = new float[halfSize+1];
   for (int i = 0; i<= halfSize; i++)
      tmp[i] = exp((float(i*i)/scale2));

   int endSize = int(ceil(scale*5.0f)-halfSize);
   for (int i = 1; i< endSize; i++)
      tmp[halfSize-i] += exp((float((i+halfSize)*(i+halfSize))/scale2));
      
   for (int i=0; i<=halfSize; i++)
      for (int j=0; j<=halfSize; j++)
      {
         mask.at<float>   ( i+halfSize,-j+halfSize) = 
            mask.at<float>(-i+halfSize, j+halfSize) = 
            mask.at<float>( i+halfSize, j+halfSize) = 
            mask.at<float>(-i+halfSize,-j+halfSize) = tmp[i]*tmp[j];
      }
   delete [] tmp;
}

void computeCircularGaussMask(Mat &mask)
{
   int size = mask.cols;
   int halfSize = size >> 1;
   float r2 = float(halfSize * halfSize);
   float sigma2 = 0.9f*r2;
   // float sigma  = float(halfSize)/3.0f;
   // float sigma2 = 2*sigma*sigma;
   float disq;
   float *mp = mask.ptr<float>(0);
   for(int i=0;i<mask.rows;i++)
      for(int j=0;j<mask.cols;j++)
      {
         disq = float((i-halfSize)*(i-halfSize)+(j-halfSize)*(j-halfSize));
         *mp++ = (disq < r2) ? exp(- disq / sigma2) : 0;
      }
}  

void invSqrt(float &a, float &b, float &c, float &l1, float &l2)
{
   double t, r;
   if (b != 0)
   {
      r = double(c-a)/(2*b);
      if (r>=0) t = 1.0/(r+::sqrt(1+r*r)); else t = -1.0/(-r+::sqrt(1+r*r));
      r = 1.0/::sqrt(1+t*t); /* c */
      t = t*r;               /* s */
   } else {
      r = 1;
      t = 0;
   }
   double x,z,d;
      
   x = 1.0/sqrt(r*r*a-2*r*t*b+t*t*c);
   z = 1.0/sqrt(t*t*a+2*r*t*b+r*r*c);
      
   d = sqrt(x*z);
   x /= d; z /= d;
   // let l1 be the greater eigenvalue
   if (x < z) { l1 = float(z); l2 = float(x); } else { l1 = float(x); l2 = float(z); }
   // output square root
   a = float( r*r*x+t*t*z);
   b = float(-r*t*x+t*r*z);
   c = float( t*t*x+r*r*z);
}
   
bool getEigenvalues(float a, float b, float c, float d, float &l1, float &l2)
{
   float trace = a+d;
   float delta1 = (trace*trace-4*(a*d-b*c));      
   if (delta1 < 0)
      return false;
   float delta = sqrt(delta1);
      
   l1 = (trace+delta)/2.0f;
   l2 = (trace-delta)/2.0f;
   return true;
}

// check if we are not too close to boundary of the image/
bool interpolateCheckBorders(const Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, const Mat &res)
{         
   const int width = im.cols-2;
   const int height = im.rows-2;
   const int halfWidth  = res.cols >> 1;
   const int halfHeight = res.rows >> 1;
   float x[4]; x[0] = -halfWidth;  x[1] = -halfWidth;  x[2] = +halfWidth;  x[3] = +halfWidth;
   float y[4]; y[0] = -halfHeight; y[1] = +halfHeight; y[2] = -halfHeight; y[3] = +halfHeight;
   for (int i=0; i<4; i++)
   {
      float imx = ofsx + x[i]*a11 + y[i]*a12;
      float imy = ofsy + x[i]*a21 + y[i]*a22;
      if (floor(imx) <= 0 || floor(imy) <= 0 || ceil(imx) >= width || ceil(imy) >= height)
         return true;
   }
   return false;
}

bool interpolate(const Mat &im, float ofsx, float ofsy, float a11, float a12, float a21, float a22, Mat &res)
{         
   bool ret = false;
   // input size (-1 for the safe bilinear interpolation)
   const int width = im.cols-1;
   const int height = im.rows-1;
   // output size
   const int halfWidth  = res.cols >> 1;
   const int halfHeight = res.rows >> 1;
   float *out = res.ptr<float>(0);
   for (int j=-halfHeight; j<=halfHeight; ++j)
   {
      const float rx = ofsx + j * a12;
      const float ry = ofsy + j * a22;
      for(int i=-halfWidth; i<=halfWidth; ++i)
      {
         float wx = rx + i * a11;
         float wy = ry + i * a21;
         const int x = (int) floor(wx);
         const int y = (int) floor(wy);
         if (x >= 0 && y >= 0 && x < width && y < height)
         {
            // compute weights
            wx -= x; wy -= y;
            // bilinear interpolation
            *out++ = 
               (1.0f - wy) * ((1.0f - wx) * im.at<float>(y,x)   + wx * im.at<float>(y,x+1)) +
               (       wy) * ((1.0f - wx) * im.at<float>(y+1,x) + wx * im.at<float>(y+1,x+1));
         } else {
            *out++ = 0;
            ret =  true; // touching boundary of the input            
         }
      }
   }
   return ret;
}

void photometricallyNormalize(Mat &image, const Mat &binaryMask, float &sum, float &var)
{   
   const int width = image.cols;
   const int height = image.rows;
   sum=0;
   float gsum=0; 

   for (int j=0; j < height; j++)
      for (int i=0; i < width; i++)
         if (binaryMask.at<float>(j,i)>0)
         {
            sum += image.at<float>(j,i); 
            gsum ++;
         }
   sum = sum / gsum;
         
   var=0;
   for (int j=0; j < height; j++)
      for (int i=0; i < width; i++)
         if (binaryMask.at<float>(j,i)>0)
            var += (sum - image.at<float>(j,i))*(sum - image.at<float>(j,i));

   var = ::sqrt(var / gsum);
   if (var < 0.0001)
      // if variance is too low, don't do anything
      return;

   float fac = 50.0f/var;   
   for (int j=0; j < height; j++)
      for (int i=0; i < width; i++)
      { 
         image.at<float>(j,i) = 128 + fac * (image.at<float>(j,i) - sum);
         if (image.at<float>(j,i) > 255) image.at<float>(j,i)=255;
         if (image.at<float>(j,i) < 0)   image.at<float>(j,i)=0;
      }
}

Mat gaussianBlur(const Mat input, float sigma)
{
   Mat ret(input.rows, input.cols, input.type());
   int size = (int)(2.0 * 3.0 * sigma + 1.0); if (size % 2 == 0) size++;      
   GaussianBlur(input, ret, Size(size, size), sigma, sigma, BORDER_REPLICATE);
   return ret;
}

void gaussianBlurInplace(Mat &inplace, float sigma)
{
   int size = (int)(2.0 * 3.0 * sigma + 1.0); if (size % 2 == 0) size++;      
   GaussianBlur(inplace, inplace, Size(size, size), sigma, sigma, BORDER_REPLICATE);
}

Mat doubleImage(const Mat &input)
{
   Mat n(input.rows*2, input.cols*2, input.type());
   const float *in = input.ptr<float>(0);
   
   for (int r = 0; r < input.rows-1; r++)
      for (int c = 0; c < input.cols-1; c++) 
      {
         const int r2 = r << 1; 
         const int c2 = c << 1;
         n.at<float>(r2,c2)     = in[0];
         n.at<float>(r2+1,c2)   = 0.5f *(in[0]+in[input.step]);
         n.at<float>(r2,c2+1)   = 0.5f *(in[0]+in[1]);
         n.at<float>(r2+1,c2+1) = 0.25f*(in[0]+in[1]+in[input.step]+in[input.step+1]);
         ++in;
      }
   for (int r = 0; r < input.rows-1; r++)
   {
      const int r2 = r << 1; 
      const int c2 = (input.cols-1) << 1;
      n.at<float>(r2,c2)   = input.at<float>(r,input.cols-1);
      n.at<float>(r2+1,c2) = 0.5f*(input.at<float>(r,input.cols-1) + input.at<float>(r+1,input.cols-1));
   }
   for (int c = 0; c < input.cols - 1; c++) 
   {
      const int r2 = (input.rows-1) << 1; 
      const int c2 = c << 1;
      n.at<float>(r2,c2)   = input.at<float>(input.rows-1,c);
      n.at<float>(r2,c2+1) = 0.5f*(input.at<float>(input.rows-1,c) + input.at<float>(input.rows-1,c+1));
   }
   n.at<float>(n.rows-1, n.cols-1) = n.at<float>(input.rows-1, input.cols-1);
   return n;
}

Mat halfImage(const Mat &input)
{
   Mat n(input.rows/2, input.cols/2, input.type());
   float *out = n.ptr<float>(0);
   for (int r = 0, ri = 0; r < n.rows; r++, ri += 2)
      for (int c = 0, ci = 0; c < n.cols; c++, ci += 2)
         *out++ = input.at<float>(ri,ci);
   return n;
}
