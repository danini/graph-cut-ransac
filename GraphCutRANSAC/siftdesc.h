/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 * 
 */

// The SIFT descriptor is subject to US Patent 6,711,293

#ifndef __SIFTDESC_H__
#define __SIFTDESC_H__

#include <vector>
#include <cv.h>
#include "helpers.h"

struct SIFTDescriptorParams
{
   int spatialBins;
   int orientationBins;
   float maxBinValue;
   int patchSize;
   SIFTDescriptorParams()
      {
         spatialBins = 4;
         orientationBins = 8;
         maxBinValue = 0.2f;
         patchSize = 41;
      } 
};


struct SIFTDescriptor
{

public:
   // top level interface
   SIFTDescriptor(const SIFTDescriptorParams &par) :
      mask(par.patchSize, par.patchSize, CV_32FC1),
      grad(par.patchSize, par.patchSize, CV_32FC1), 
      ori(par.patchSize, par.patchSize, CV_32FC1)
      {
         this->par = par;
         vec.resize(par.spatialBins * par.spatialBins * par.orientationBins);
         computeCircularGaussMask(mask);
         precomputeBinsAndWeights();
      }
   
   void computeSiftDescriptor(cv::Mat &patch);

public:
   std::vector<float> vec;

private:
   // helper functions
   
float normalize();
   void sample();
   void samplePatch();
   void precomputeBinsAndWeights();

private:
   SIFTDescriptorParams par;
   cv::Mat mask, grad, ori;
   std::vector<int> precomp_bins;
   std::vector<float> precomp_weights;
   int *bin0, *bin1;
   float *w0, *w1;
};

#endif //__SIFTDESC_H__
