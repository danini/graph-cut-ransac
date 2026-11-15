// Regression test: the samples of ProsacSampler must not repeat a point.
// The first sample (after construction and after reset()) used to draw its
// first m - 1 points from [0, m - 1] and then add point m - 1, so it could
// contain point m - 1 twice. Exits with 0 if no sample repeats a point.
#include <algorithm>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "GCoptimization.h" // defines OLGA_INLINE
#include "samplers/prosac_sampler.h"

int main()
{
	const size_t point_number = 20, iterations = 200, trials = 100;
	const cv::Mat points(point_number, 4, CV_64F, cv::Scalar(0));
	const std::vector<size_t> pool; // ProsacSampler::sample doesn't use it
	int failures = 0;
	for (size_t sample_size : {1, 2, 4, 7}) // 1: the one-point sampler of ProgressiveNapsacSampler
	{
		std::vector<size_t> sample(sample_size);
		for (size_t trial = 0; trial < trials; ++trial)
		{
			gcransac::sampler::ProsacSampler sampler(&points, sample_size);
			for (int pass = 0; pass < 2; ++pass) // after construction, then after reset()
			{
				for (size_t k = 0; k < iterations; ++k)
				{
					sampler.sample(pool, sample.data(), sample_size);
					std::vector<size_t> sorted = sample;
					std::sort(sorted.begin(), sorted.end());
					if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end() ||
						sorted.back() >= point_number)
					{
						if (failures++ < 5)
						{
							std::printf("FAIL sample size %zu, %s, sample %zu:", sample_size,
								pass == 0 ? "after construction" : "after reset()", k + 1);
							for (size_t i : sample)
								std::printf(" %zu", i);
							std::printf("\n");
						}
					}
				}
				sampler.reset();
			}
		}
	}
	if (failures)
		std::printf("%d samples repeat a point\n", failures);
	else
		std::printf("ok: no sample repeats a point\n");
	return failures ? 1 : 0;
}
