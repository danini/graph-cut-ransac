#pragma once

#include <random>
#include <algorithm>

class UniformRandomGenerator
{
protected:
	std::mt19937 generator;
	std::uniform_int_distribution<int> generate;

public:
	UniformRandomGenerator() {
		std::random_device rand_dev;
		generator = std::mt19937(rand_dev());
	}

	~UniformRandomGenerator() {

	}

	inline int get_random_number() {
		return generate(generator);
	}

	void reset_generator(int min_range_,
		int max_range_) {
		generate = std::uniform_int_distribution<int>(min_range_, max_range_);
	}

	inline void generate_unique_random_set(int * sample_,
		unsigned int sample_size_)
	{
		for (auto i = 0; i < sample_size_; i++)
		{
			sample_[i] = generate(generator);
			for (int j = i - 1; j >= 0; j--) {
				if (sample_[i] == sample_[j]) {
					i--;
					break;
				}
			}
		}
	}

	inline void generate_unique_random_set(int * sample_,
		unsigned int sample_size_,
		unsigned int max_) {
		reset_generator(0, max_);
		for (auto i = 0; i < sample_size_; i++) {
			sample_[i] = generate(generator);
			for (int j = i - 1; j >= 0; j--) {
				if (sample_[i] == sample_[j]) {
					i--;
					break;
				}
			}
		}
	}
};