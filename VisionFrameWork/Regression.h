#pragma once

#include "ceres/ceres.h"

using ceres::AutoDiffCostFunction;
using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::HuberLoss;
using ceres::Solve;

template <typename T>
class Regression
{
public:
	enum REG_NORM {L1, L2, L2_TRUNCATED};
	enum DISTANCE_FUNCTION { Eucledian, SymmetricEpipolar, Sampson };
	enum MODEL_TYPE {LINE, CIRCLE, PLANE, CYLINDER, HOMOGRAPHY, FUNDAMENTAL};

	struct CostFunctor
	{
		Mat const * const points;
		int dimensions;
		REG_NORM reg_norm;
		T threshold;
		T sqr_threshold;
		DISTANCE_FUNCTION distance_type;
		MODEL_TYPE model_type;

		CostFunctor(Mat const * const _points, MODEL_TYPE _model_type, REG_NORM _reg_norm, DISTANCE_FUNCTION _distance_type, T _threshold = 0) : points(_points),
			model_type(_model_type),
			dimensions(_points->cols),
			distance_type(_distance_type),
			reg_norm(_reg_norm),
			threshold(_threshold),
			sqr_threshold(_threshold * _threshold)
		{
		}

		template <typename S> bool operator()(const S* const x, S* residual) const
		{
			S distance = S(0.0);
			S* model;
			S dist;

			switch (model_type)
			{
			case MODEL_TYPE::LINE:
				model = new S[3];

				dist = sqrt(x[0] * x[0] + x[1] * x[1]);

				model[0] = x[0] / dist;
				model[1] = x[1] / dist;
				model[2] = x[2];

				for (int i = 0; i < points->rows; ++i)
					distance += LinearDistance<S>(&points->row(i), model);
				break;
			
			case MODEL_TYPE::PLANE:
				model = new S[dimensions + 1];
				if (dimensions == 2)
				{
					//S dist = sqrt(x[0] * x[0] + x[1] * x[1]);

					model[0] = cos(x[0]);
					model[1] = sin(x[0]);
					model[2] = x[1];

				}
				else if (dimensions == 3)
				{
					S theta = x[0];
					S sigma = x[1];
					model[0] = cos(theta) * sin(sigma);
					model[1] = sin(theta) * sin(sigma);
					model[2] = cos(sigma);
					model[3] = x[2];
				}

				for (int i = 0; i < points->rows; ++i)
					distance += LinearDistance<S>(&points->row(i), model);
				break;
			case MODEL_TYPE::CYLINDER:
				for (int i = 0; i < points->rows; ++i)
					distance +=  CircleDistance<S>(&points->row(i), x);
				break;
			case MODEL_TYPE::CIRCLE:
				break;
			case MODEL_TYPE::HOMOGRAPHY:
				break;
			case MODEL_TYPE::FUNDAMENTAL:
				if (distance_type == DISTANCE_FUNCTION::SymmetricEpipolar)
				{

				}
				else if (distance_type == DISTANCE_FUNCTION::Sampson)
				{
					for (int i = 0; i < points->rows; ++i)
						distance += SampsonDistance<S>(&points->row(i), x);
				}

				break;
			}

			residual[0] = distance;

			return true;
		}

		template <typename S> S SampsonDistance(Mat const * const point, const S* const x) const
		{
			S x1 = S(point->at<T>(0));
			S y1 = S(point->at<T>(1));
			S x2 = S(point->at<T>(3));
			S y2 = S(point->at<T>(4));

			S a1 = x2 * x[0] + y2 * x[3] + x[6];
			S a2 = x2 * x[1] + y2 * x[4] + x[7];
			S a3 = x2 * x[2] + y2 * x[5] + x[8];

			S b1 = x1 * x[0] + y1 * x[1] + x[2];
			S b2 = x1 * x[3] + y1 * x[4] + x[5];
			S b3 = x1 * x[6] + y1 * x[7] + x[8];

			S c = x1 * b1 + x2 * b2 + b3;
			S d = a1*a1 + a2*a2 + b1*b1 + b2*b2;
			S distance = c / d;
			
			if (reg_norm == REG_NORM::L1)
				return abs(distance);
			if (reg_norm == REG_NORM::L2)
				return distance * distance;
			if (reg_norm == REG_NORM::L2_TRUNCATED)
				return MIN(distance * distance, S(sqr_threshold));
		}

		template <typename S> S LinearDistance(Mat const * const point, const S* const x) const
		{
			S distance = S(0.0);
			for (int d = 0; d < dimensions; ++d)
				distance += S(point->at<T>(d) * x[d]);
			distance += S(x[dimensions]);

			if (reg_norm == REG_NORM::L1)
				return abs(distance);
			if (reg_norm == REG_NORM::L2)
				return distance * distance;
			if (reg_norm == REG_NORM::L2_TRUNCATED)
				return MIN(distance * distance, S(sqr_threshold));

			return distance;
		}

		template <typename S> S CircleDistance(Mat const * const point, const S* const x) const
		{
			S dx = S(points->at<T>(0) - x[0]);
			S dy = S(points->at<T>(1) - x[1]);

			S distFromCenter = sqrt(dx*dx + dy*dy);
			S distance = distFromCenter - x[2];
			for (int d = 0; d < dimensions; ++d)
				distance += S(point->at<T>(d) * x[d]);
			distance += S(x[dimensions]);

			if (reg_norm == REG_NORM::L1)
				return abs(distance);
			if (reg_norm == REG_NORM::L2)
				return distance * distance;
			if (reg_norm == REG_NORM::L2_TRUNCATED)
				return MIN(distance * distance, S(sqr_threshold));

			return distance;
		}
	};

	Regression();
	~Regression() {}

	void Run(const Mat const * _points, const Mat const &_initial_parameters, Mat &_refined_parameters, MODEL_TYPE model_type, REG_NORM _reg_norm = REG_NORM::L1, DISTANCE_FUNCTION _distance_function = DISTANCE_FUNCTION::Eucledian, T threshold = T(1.0));
};

template <typename T>
Regression<T>::Regression()
{
}

template <typename T>
void Regression<T>::Run(const Mat const * _points, const Mat const &_initial_parameters, Mat &_refined_parameters, MODEL_TYPE model_type, REG_NORM _reg_norm, DISTANCE_FUNCTION _distance_function, T threshold)
{
	Problem problem;

	// Initialize parameters
	T *params = new T[_initial_parameters.rows * _initial_parameters.cols];
	for (int i = 0; i < _initial_parameters.rows; ++i)
		for (int j = 0; j < _initial_parameters.cols; ++j)
			params[i * _initial_parameters.cols + j] = (T)_initial_parameters.at<float>(i, j);

	CostFunction* cost_function = NULL;

	switch (model_type)
	{
	case MODEL_TYPE::LINE:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	case MODEL_TYPE::PLANE:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	case MODEL_TYPE::CYLINDER:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 9>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	case MODEL_TYPE::CIRCLE:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	case MODEL_TYPE::HOMOGRAPHY:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 9>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	case MODEL_TYPE::FUNDAMENTAL:
		cost_function = new AutoDiffCostFunction<CostFunctor, 1, 9>(new CostFunctor(_points, model_type, _reg_norm, _distance_function, threshold));
		break;
	}
	
	if (cost_function == NULL)
		return;

	problem.AddResidualBlock(cost_function, new HuberLoss(1.0), params);

	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = false;
	options.function_tolerance = 1e-100;
	options.gradient_tolerance = 1e-100;
	options.max_num_iterations = 100;
	options.preconditioner_type = ceres::JACOBI;

	Solver::Summary summary;
	Solve(options, &problem, &summary);

	_refined_parameters = Mat(_initial_parameters.size(), _initial_parameters.type());
	for (int i = 0; i < _initial_parameters.rows; ++i)
		for (int j = 0; j < _initial_parameters.cols; ++j)
			_refined_parameters.at<float>(i, j) = (float)params[i * _initial_parameters.cols + j];

	std::cout << summary.BriefReport() << "\n";
}

