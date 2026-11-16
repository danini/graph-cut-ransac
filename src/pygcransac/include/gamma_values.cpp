#pragma once

// Lookup tables of the incomplete gamma functions used by MAGSAC++ for 4 degrees of freedom,
// computed once, at static initialization. Entry i of each table is:
// - stored_complete_gamma_values[i]: the upper incomplete gamma Gamma(3/2, i / 1e4), i = 0, ..., 100000;
// - stored_lower_incomplete_gamma_values[i]: the lower incomplete gamma gamma(5/2, i / 1e4), i = 0, ..., 100000;
// - stored_gamma_values[i]: Gamma(3/2, i / 1e3), i = 0, ..., 38682 (up to the first value below 1e-16).
//   Its step is 1e-3, not 1 / precision_of_stored_gammas.
// These half-integer orders have closed forms:
//   Gamma(3/2, x) = sqrt(x) exp(-x) + sqrt(pi) / 2 erfc(sqrt(x)),
//   gamma(5/2, x) = 3 sqrt(pi) / 4 erf(sqrt(x)) - sqrt(x) exp(-x) (x + 3/2).
// This file is included as a header, by scoring_function.h and by MAGSAC's magsac.h.

#include <cmath>
#include <cstddef>
#include <vector>

constexpr unsigned int stored_gamma_number = 36000;
constexpr unsigned int stored_incomplete_gamma_number = 100000;
constexpr double precision_of_stored_gammas = 1e4;
constexpr double precision_of_stored_incomplete_gammas = 1e4;

namespace gamma_values_detail
{
	constexpr double sqrt_pi = 1.7724538509055160273;

	inline double upper_gamma_3_2(double x)
	{
		return std::sqrt(x) * std::exp(-x) + sqrt_pi / 2 * std::erfc(std::sqrt(x));
	}

	inline double lower_gamma_5_2(double x)
	{
		// For small x, the closed form loses digits to cancellation: use the series
		// gamma(a, x) = x^a exp(-x) sum_n x^n / (a (a + 1) ... (a + n)) instead.
		if (x < 1)
		{
			double term = 1 / 2.5, sum = term;
			for (int n = 1; n < 20; ++n)
			{
				term *= x / (2.5 + n);
				sum += term;
			}
			return x * x * std::sqrt(x) * std::exp(-x) * sum;
		}
		return 3 * sqrt_pi / 4 * std::erf(std::sqrt(x)) - std::sqrt(x) * std::exp(-x) * (x + 1.5);
	}

	// f(i / precision) for i = 0, ..., size - 1. The tables are never destroyed, so that a thread
	// still scoring while the program exits doesn't read freed memory.
	inline std::vector<double> *table(double (*f)(double), std::size_t size, double precision)
	{
		auto *values = new std::vector<double>(size);
		for (std::size_t i = 0; i < size; ++i)
			(*values)[i] = f(static_cast<double>(i) / precision);
		return values;
	}
}

inline const std::vector<double> &stored_complete_gamma_values = *gamma_values_detail::table(
	gamma_values_detail::upper_gamma_3_2, stored_incomplete_gamma_number + 1, precision_of_stored_incomplete_gammas);
inline const std::vector<double> &stored_lower_incomplete_gamma_values = *gamma_values_detail::table(
	gamma_values_detail::lower_gamma_5_2, stored_incomplete_gamma_number + 1, precision_of_stored_incomplete_gammas);
inline const std::vector<double> &stored_gamma_values = *gamma_values_detail::table(
	gamma_values_detail::upper_gamma_3_2, 38683, 1e3);
