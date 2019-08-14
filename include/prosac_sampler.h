// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SOLVERS_PROSAC_SAMPLER_H_
#define THEIA_SOLVERS_PROSAC_SAMPLER_H_

#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

#include "sampler.h"
#include "uniform_random_generator.h"

namespace theia
{
	namespace
	{
		std::default_random_engine util_generator;
	}  // namespace

	// Prosac sampler used for PROSAC implemented according to "cv::Matching with PROSAC
	// - Progressive Sampling Consensus" by Chum and cv::Matas.
	class ProsacSampler : public Sampler < cv::Mat, size_t >
	{
	public:

		// Get a random int between lower_ and upper_ (inclusive).
		template<typename _ValueType>
		_ValueType randomNumber(
			_ValueType lower_,
			_ValueType upper_)
		{
			std::uniform_int_distribution<_ValueType> distribution(lower_, upper_);
			return distribution(util_generator);
		}

		explicit ProsacSampler(const cv::Mat const *container_,
			const size_t sample_size_) : 
			sample_size(sample_size_),
			Sampler(container_)
		{
			initialize(container);
		}

		~ProsacSampler() {}

		bool initialize(const cv::Mat const *container_)
		{
			ransac_convergence_iterations = 100000;
			kth_sample_number = 1;
			
			unsigned seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
			util_generator.seed(seed);
			return true;
		}

		// Set the sample such that you are sampling the kth prosac sample (Eq. 6).
		void setSampleNumber(int k)
		{
			kth_sample_number = k;
		}

		// Samples the input variable data and fills the std::vector subset with the prosac
		// samples.
		// NOTE: This assumes that data is in sorted order by quality where data[i] is
		// of higher quality than data[j] for all i < j.
		bool sample(const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_)
		{
			// Set t_n according to the PROSAC paper's recommendation.
			double t_n = ransac_convergence_iterations;
			const int point_number = static_cast<int>(pool_.size());
			int n = this->sample_size;
			// From Equations leading up to Eq 3 in Chum et al.
			for (auto i = 0; i < this->sample_size; i++)
			{
				t_n *= static_cast<double>(n - i) / (point_number - i);
			}

			double t_n_prime = 1.0;
			// Choose min n such that T_n_prime >= t (Eq. 5).
			for (auto t = 1; t <= kth_sample_number; t++)
			{
				if (t > t_n_prime && n < point_number)
				{
					double t_n_plus1 =
						(t_n * (n + 1.0)) / (n + 1.0 - this->sample_size);
					t_n_prime += ceil(t_n_plus1 - t_n);
					t_n = t_n_plus1;
					n++;
				}
			}

			if (t_n_prime < kth_sample_number)
			{
				// Randomly sample m data points from the top n data points.
				std::vector<int> random_numbers;
				for (auto i = 0; i < this->sample_size; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = randomNumber<size_t>(0, n - 1))) !=
						random_numbers.end())
					{
					}

					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset_[i] = pool_[rand_number];
					//subset->emplace_back(data[rand_number]);
				}
			}
			else
			{
				std::vector<size_t> random_numbers;
				// Randomly sample m-1 data points from the top n-1 data points.
				for (auto i = 0; i < this->sample_size - 1; i++)
				{
					// Generate a random number that has not already been used.
					int rand_number;
					while (std::find(random_numbers.begin(), random_numbers.end(),
						(rand_number = randomNumber<size_t>(0, n - 2))) !=
						random_numbers.end())
					{
					}
					random_numbers.emplace_back(rand_number);

					// Push the *unique* random index back.
					subset_[i] = pool_[rand_number];
					//subset->emplace_back(data[rand_number]);
				}
				// Make the last point from the nth position.
				subset_[this->sample_size - 1] = pool_[n];
				//subset->emplace_back(data[n]);
			}
			kth_sample_number++;
			return true;
		}
	private:
		size_t sample_size;

		// Number of iterations of PROSAC before it just acts like ransac.
		size_t ransac_convergence_iterations;

		// The kth sample of prosac sampling.
		size_t kth_sample_number;
	};

}  // namespace theia

#endif  // THEIA_SOLVERS_PROSAC_SAMPLER_H_
