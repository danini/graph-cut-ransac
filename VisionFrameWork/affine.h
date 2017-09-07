/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

#ifndef __AFFINE_H__
#define __AFFINE_H__

#include <vector>
#include <cv.h>
#include "helpers.h"

struct AffineShapeParams
{
   // number of affine shape interations
   int maxIterations;

   // convergence threshold, i.e. maximum deviation from isotropic shape at convergence
   float convergenceThreshold;

   // widht and height of the SMM mask 
   int smmWindowSize;

   // width and height of the patch
   int patchSize;

   // amount of smoothing applied to the initial level of first octave
   float initialSigma;

   // size of the measurement region (as multiple of the feature scale)
   float mrSize;

   AffineShapeParams()
      {
         maxIterations = 32;
         initialSigma = 1.6f;
         convergenceThreshold = 0.001;
         patchSize = 41;
         smmWindowSize = 19;
         mrSize = 3.0f*sqrt(3.0f);
      }
};

struct AffineShapeCallback
{
   virtual void onAffineShapeFound(
      const cv::Mat &blur,     // corresponding scale level
      float x, float y,     // subpixel, image coordinates
      float s,              // scale
      float pixelDistance,  // distance between pixels in provided blured image
      float a11, float a12, // affine shape matrix 
      float a21, float a22, 
      int type, float response, int iters) = 0;
};

struct AffineShape
{
public:   
   AffineShape(const AffineShapeParams &par) : 
      patch(par.patchSize, par.patchSize, CV_32FC1),
      mask(par.smmWindowSize, par.smmWindowSize, CV_32FC1), 
      img(par.smmWindowSize, par.smmWindowSize, CV_32FC1), 
      fx(par.smmWindowSize, par.smmWindowSize, CV_32FC1), 
      fy(par.smmWindowSize, par.smmWindowSize, CV_32FC1)
      {                     
         this->par = par;
         computeGaussMask(mask);
         affineShapeCallback = 0;
         fx = cv::Scalar(0);
         fy = cv::Scalar(0);
      }
   
   ~AffineShape()
      {
      }
   
   // computes affine shape 
   bool findAffineShape(const cv::Mat &blur, float x, float y, float s, float pixelDistance, int type, float response);   

   // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times
   bool normalizeAffine(const cv::Mat &img, float x, float y, float s, float a11, float a12, float a21, float a22);

   void setAffineShapeCallback(AffineShapeCallback *callback)
      {
         affineShapeCallback = callback;
      }

public:
   cv::Mat patch;

protected:
   AffineShapeParams par;

private:
   AffineShapeCallback *affineShapeCallback;
   std::vector<unsigned char> workspace;
   cv::Mat mask, img, fx, fy;
};

#endif // __AFFINE_H__
