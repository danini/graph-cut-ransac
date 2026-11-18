// Copyright (C) 2019 Czech Technical University.
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
//     * Neither the name of Czech Technical University nor the
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
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <random>
#include <algorithm>
#include <optional>
#include <cstdint>

namespace gcransac
{
	namespace utils
	{
		template <typename _ValueType>
		class UniformRandomGenerator
		{
		protected:
			std::mt19937 generator;
			std::uniform_int_distribution<_ValueType> generate;

			// The seed set by setGlobalSeed, if any, and the number of generators
			// created since in this thread.
			inline static thread_local std::optional<std::uint64_t> global_seed;
			inline static thread_local std::uint64_t generator_count = 0;

		public:
			// Seeds the generators created afterwards in the calling thread from
			// seed and their creation index (std::seed_seq), instead of from
			// std::random_device, until clearGlobalSeed(). Generators created in
			// other threads aren't affected (the seed is thread_local). Each
			// generator of a run gets its own stream (P-NAPSAC has three), and the
			// runs of different seeds don't share streams.
			static void setGlobalSeed(std::uint64_t seed) { global_seed = seed; generator_count = 0; }
			// Seeds the generators created afterwards in the calling thread from
			// std::random_device again.
			static void clearGlobalSeed() { global_seed.reset(); }

			UniformRandomGenerator() {
				if (global_seed)
				{
					const std::uint64_t index = generator_count++;
					std::seed_seq seeds{
						static_cast<std::uint32_t>(*global_seed), static_cast<std::uint32_t>(*global_seed >> 32),
						static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(index >> 32) };
					generator = std::mt19937(seeds);
				}
				else
				{
					std::random_device rand_dev;
					generator = std::mt19937(rand_dev());
				}
			}

			~UniformRandomGenerator() {

			}

			std::mt19937 &getGenerator()
			{
				return generator;
			}

			OLGA_INLINE int getRandomNumber() {
				return generate(generator);
			}

			OLGA_INLINE void resetGenerator(
				_ValueType min_range_,
				_ValueType max_range_) {
				generate = std::uniform_int_distribution<_ValueType>(min_range_, max_range_);
			}

			OLGA_INLINE void generateUniqueRandomSet(_ValueType * sample_,
				size_t sample_size_)
			{
				for (size_t i = 0; i < sample_size_; i++)
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

			OLGA_INLINE void generateUniqueRandomSet(_ValueType * sample_,
				size_t sample_size_,
				_ValueType max_) {
				resetGenerator(0, max_);
				for (size_t i = 0; i < sample_size_; i++) {
					sample_[i] = generate(generator);
					for (int j = i - 1; j >= 0; j--) {
						if (sample_[i] == sample_[j]) {
							i--;
							break;
						}
					}
				}
			}

			OLGA_INLINE void generateUniqueRandomSet(_ValueType * sample_,
				size_t sample_size_,
				_ValueType max_,
				_ValueType to_skip_) {
				resetGenerator(0, max_);
				for (size_t i = 0; i < sample_size_; i++) {
					sample_[i] = generate(generator);
					if (sample_[i] == to_skip_) {
						i--;
						continue;
					}

					for (int j = i - 1; j >= 0; j--) {
						if (sample_[i] == sample_[j]) {
							i--;
							break;
						}
					}
				}
			}
		};
	}
}