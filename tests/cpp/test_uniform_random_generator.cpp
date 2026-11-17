// Tests the seeding of UniformRandomGenerator (setGlobalSeed / clearGlobalSeed).
#include <cstdio>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>

#include "GCoptimization.h"  // defines OLGA_INLINE, which the samplers use
#include "samplers/progressive_napsac_sampler.h"
#include "uniform_random_generator.h"

using Generator = gcransac::utils::UniformRandomGenerator<size_t>;

// The first numbers of a copy of the generator (the generator itself is unchanged)
static std::vector<unsigned> stream(std::mt19937 generator)
{
	std::vector<unsigned> numbers(16);
	for (unsigned &number : numbers)
		number = generator();
	return numbers;
}

// Exposes P-NAPSAC's own generator, created after those of its two nested PROSAC samplers
struct ProgressiveNapsac : gcransac::sampler::ProgressiveNapsacSampler<4>
{
	using ProgressiveNapsacSampler::ProgressiveNapsacSampler;
	std::vector<unsigned> ownStream() { return stream(random_generator->getGenerator()); }
};

static int failures = 0;

static void check(bool condition, const char *what)
{
	if (!condition)
	{
		std::fprintf(stderr, "FAILED: %s\n", what);
		++failures;
	}
}

int main()
{
	// Two generators created after one seed get different streams, and the same seed gives them again
	Generator::setGlobalSeed(42);
	Generator a, b;
	Generator::setGlobalSeed(42);
	Generator a2, b2;
	check(stream(a.getGenerator()) != stream(b.getGenerator()), "generators created after one seed differ");
	check(stream(a.getGenerator()) == stream(a2.getGenerator()) &&
		stream(b.getGenerator()) == stream(b2.getGenerator()), "the same seed gives the same generators");

	// Consecutive seeds don't share streams: the second generator of seed 42 isn't the first of seed 43
	Generator::setGlobalSeed(43);
	Generator first43;
	check(stream(b.getGenerator()) != stream(first43.getGenerator()), "consecutive seeds give distinct streams");

	// P-NAPSAC: its own generator doesn't repeat the stream of its first nested PROSAC sampler,
	// the first generator created after the seed
	cv::Mat points(100, 4, CV_64F);
	cv::randu(points, 0, 100);
	const std::vector<size_t> layers = { 16, 8, 4, 2 };
	const std::vector<double> sizes = { 100, 100, 100, 100 };
	Generator::setGlobalSeed(7);
	Generator first7;
	Generator::setGlobalSeed(7);
	ProgressiveNapsac sampler(&points, layers, 4, sizes);
	Generator::setGlobalSeed(7);
	ProgressiveNapsac sampler2(&points, layers, 4, sizes);
	check(sampler.isInitialized(), "P-NAPSAC is initialized");
	check(sampler.ownStream() != stream(first7.getGenerator()), "P-NAPSAC's generators get distinct seeds");
	check(sampler.ownStream() == sampler2.ownStream(), "the same seed gives the same P-NAPSAC generator");

	// Without a seed, generators are seeded from std::random_device again
	Generator::clearGlobalSeed();
	Generator c, d;
	check(stream(c.getGenerator()) != stream(d.getGenerator()), "unseeded generators differ");

	if (failures == 0)
		std::printf("All checks passed\n");
	return failures == 0 ? 0 : 1;
}
