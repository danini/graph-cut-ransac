#pragma once

#include <opencv\cv.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2\calib3d\calib3d.hpp>
#include "MainApp.h"
#include <stdio.h>
#include <vector>
#include "pso.h"
#include <time.h>
#include <algorithm>
using namespace cv;

template <typename T>
class objective_function
{
public:
	const double* weights;
	objective_function(int dimensions) : N(dimensions)
	{
		// initialize pso settings
		pso_set_default_settings(settings);
		settings.size = 30;
		//settings.nhood_strategy = PSO_NHOOD_RING; // Nem grid (random) elosztás esetén ez szokott jó lenni
		settings.nhood_strategy = PSO_NHOOD_RANDOM; // Ha pedig grid eloszltás van, akkor ez a zsír
		settings.nhood_size = (1 << dimensions); // RANDOM eseten fontos(!)

		settings.w_strategy = PSO_W_LIN_DEC;
		settings.dim = dimensions;
		settings.steps = 100;

		// Random seed kell a swarm es a random neighbourhood randomsagaihoz
		settings.seed = long(time(0));
		settings.clamp_pos = 1;

		Centering1 = Mat::eye(3, 3, CV_64F);
		Centering2 = Mat::eye(3, 3, CV_64F);
		invCentering1 = Mat::eye(3, 3, CV_64F);
		invCentering2 = Mat::eye(3, 3, CV_64F);
	}

	void set_constraint_settings(vector<double> low, vector<double> high, double goal_treshold, int num_particles) {
		settings.lo = low;
		settings.hi = high;
		settings.goal = goal_treshold;
		settings.size = num_particles;
	}

	// The algorithm swaps to L-BFGS num. opt. if this criterion is true
	void set_swapping_criterion(double treshold, unsigned window)
	{
		settings.stopping_criterion_treshold = treshold;
		settings.stopping_criterion_window = window;
	}

protected:
	int N;
	std::vector<T> m_x;
	T fx;
	pso_settings_t settings; // Particle Swarm Optimization settings
	
	Mat *origPatch1;
	Mat *origPatch2;
	Mat *origPatch1Big;
	Mat *origPatch2Big;

	Mat epiline2;

	/*FOR CORRELATION*/
	Mat Centering1;
	Mat Centering2;
	Mat invCentering1;
	Mat invCentering2;
	double s1s2;
	Mat cross0, cross1, cross2, cross3, cross4;

	int run_pso()
	{
		assert(settings.lo.size() == settings.hi.size());
		for(unsigned i = 0; i < settings.lo.size(); ++i)
			assert(settings.lo[i] <= settings.hi[i]);

		pso_result_t solution;
		solution.gbest.resize(settings.dim);

		// run
		pso_solve(&pso_evaluate, this, &solution, settings);

		// prepare data for bfgs
		fx = solution.error;
		std::copy(solution.gbest.begin(), solution.gbest.end(), m_x.begin());
		return solution.stopping_criterion;
	}

	int run_exhaustive_search()
	{
		assert(settings.lo.size() == settings.hi.size());
		for(unsigned i = 0; i < settings.lo.size(); ++i)
			assert(settings.lo[i] <= settings.hi[i]);

		fx = 0;
		m_x.resize(settings.dim);
		static const int resU = 100;
		static const int resV = 100;

		double minU = settings.lo[0];
		double minV = settings.lo[1];
		double stepU = (settings.hi[0] - minU) / double(resU-1);
		double stepV = (settings.hi[1] - minV) / double(resV-1);

		Point2d maxPos(0,0);
		double maxVal = -2;
		Point2d currentPos;

		for(int i = 0; i < resU; ++i) {
			currentPos.x = minU + i * stepU;
			for(int j = 0; j < resV; ++j) {
				currentPos.y = minV + j * stepV;
				double val = evaluate(&currentPos.x, NULL, 0, 0.0);
				if(val > maxVal) {
					maxVal = val;
					maxPos = currentPos;
				}
			}
		}

		fx = maxVal;
		m_x[0] = maxPos.x;
		m_x[1] = maxPos.y;

		return 0;
	}

	int run_bfgs()
	{
		// TODO
		return 0;
	}

public:
	T GetFX() const { return fx; }

	std::vector<T> getSolution() { return m_x; }
    int run()
    {
		m_x.resize(N);

		int ret;

		ret = run_pso();
		//ret = run_exhaustive_search();
		ret |= run_bfgs(); // TODO

		return ret;
    }

	Rect_<double> setValues(int particles, Mat *_origPatch1, Mat *_origPatch2, Mat *_origPatch1Big, Mat *_origPatch2Big,
		Mat F, Mat P1, Mat P2, Mat R1, Mat R2, Vec3d T1, Vec3d T2, Vec3d referencePoint, Point2d pt1, Point2d pt2, MainApp *app, Mat eline2, double maxDistOnEpiline, const double* _weights)
	{
		origPatch1=_origPatch1;
		origPatch2=_origPatch2;
		origPatch1Big=_origPatch1Big;
		origPatch2Big=_origPatch2Big;

		epiline2 = eline2;

		weights = _weights;
		/*int M = origPatch1->size().height;
		int N = origPatch1->size().width;
		if (weights.size() != M*N) { //hehe not safe
			weights.resize(M*N);
			if(sigma < 0) {
				sigma = (M*N/4.0);
			}
			init_sqrtGaussianWeights(M,N, 1.0/sigma, &weights[0]);
		}*/

		Vec3d vp1 = T1 - referencePoint;
		Vec3d vp2 = T2 - referencePoint;

		cv::Vec2d params1 = ToSphericalNormal(vp1);
		cv::Vec2d params2 = ToSphericalNormal(vp2);
		
		static const cv::Vec2d pivec2 = cv::Vec2d(M_PI_2, M_PI_2);
		Vec2d amin = params1-pivec2;
		Vec2d amax = params1+pivec2;
		Vec2d bmin = params2-pivec2;
		Vec2d bmax = params2+pivec2;

		Vec2d rmin = rectmax(amin, bmin);
		Vec2d rmax = rectmin(amax, bmax);
		Vec2d diff = rmax - rmin;

		if(diff[0] <= 0 || diff[1] <= 0) { //TODO what the actual fuck?
			rmin = rectmin(amin, bmin); //TODO what the actual fuck?
			rmax = rectmax(amax, bmax); //TODO what the actual fuck?
		}
		/*if(diff[0] < 0 || diff[1] < 0) { //TODO what the actual fuck? / unused /
			rmin = amin;
			rmax = amax;
		}*/
		
		diff = rmax - rmin;

		// TODO: remove 
		//if (!r1.contains(params2) || !r2.contains(params1)) // analogous to ( vp1.dot(vp2) < 0 )
		//	return FAIL;

		vector<double> lo; lo.resize(3);
		vector<double> hi; hi.resize(3);
		lo[0] = rmin[0];
		hi[0] = rmax[0];
		lo[1] = rmin[1];
		hi[1] = rmax[1];
		lo[2] = -maxDistOnEpiline;
		hi[2] = maxDistOnEpiline;
		
		double portion = (diff[0]*diff[1]) / (M_PI * M_PI);
		int num = std::max( std::min( int(ceil(sqrt(particles * portion))), 10 ), 2);
		
		set_constraint_settings( lo, hi, 1e-6, num*num );
		set_swapping_criterion( 1e-9, 6 );

		/*FOR CORRELATION*/
		// Speetup GetAffine!
		{
			Centering1.at<double>(0, 2) = origPatch1->size().width / 2.0; Centering1.at<double>(1, 2) = origPatch1->size().height / 2.0;
			invCentering1.at<double>(0, 2) = -origPatch1->size().width / 2.0; invCentering1.at<double>(1, 2) = -origPatch1->size().height / 2.0;
			/*Centering2.at<double>(0, 2) = origPatch2->size().width / 2.0; Centering2.at<double>(1, 2) = origPatch2->size().height / 2.0;
			invCentering2.at<double>(0, 2) = -origPatch2->size().width / 2.0; invCentering2.at<double>(1, 2) = -origPatch2->size().height / 2.0;*/

			Mat pt3dp1 = (Mat_<double>(4, 1, CV_64F) << referencePoint[0], referencePoint[1], referencePoint[2], 1);
			double s1 = pt3dp1.dot(P1.row(2).t());
			double s2 = pt3dp1.dot(P2.row(2).t());
			s1s2 = s1/s2;

			double dxx1 = (P1.at<double>(0, 0) - P1.at<double>(2, 0) * pt1.x);
			double dxy1 = (P1.at<double>(0, 1) - P1.at<double>(2, 1) * pt1.x);
			double dxz1 = (P1.at<double>(0, 2) - P1.at<double>(2, 2) * pt1.x);
			double dyx1 = (P1.at<double>(1, 0) - P1.at<double>(2, 0) * pt1.y);
			double dyy1 = (P1.at<double>(1, 1) - P1.at<double>(2, 1) * pt1.y);
			double dyz1 = (P1.at<double>(1, 2) - P1.at<double>(2, 2) * pt1.y);
				  														  
			double dxx2 = (P2.at<double>(0, 0) - P2.at<double>(2, 0) * pt2.x);
			double dxy2 = (P2.at<double>(0, 1) - P2.at<double>(2, 1) * pt2.x);
			double dxz2 = (P2.at<double>(0, 2) - P2.at<double>(2, 2) * pt2.x);
			double dyx2 = (P2.at<double>(1, 0) - P2.at<double>(2, 0) * pt2.y);
			double dyy2 = (P2.at<double>(1, 1) - P2.at<double>(2, 1) * pt2.y);
			double dyz2 = (P2.at<double>(1, 2) - P2.at<double>(2, 2) * pt2.y);

			Mat deltaX1 = (Mat_<double>(3, 1) << dxx1, dxy1, dxz1);
			Mat deltaY1 = (Mat_<double>(3, 1) << dyx1, dyy1, dyz1);
			Mat deltaX2 = (Mat_<double>(3, 1) << dxx2, dxy2, dxz2);
			Mat deltaY2 = (Mat_<double>(3, 1) << dyx2, dyy2, dyz2);

			cross0 = deltaX1.cross(deltaY1);
			cross1 = deltaY1.cross(deltaX2);
			cross2 = deltaX2.cross(deltaX1);
			cross3 = deltaY1.cross(deltaY2);
			cross4 = deltaY2.cross(deltaX1);
			
		}

		return Rect_<double>(rmax, rmin);
	}

	void CalculateCrossVectors(Point2d pt1, Point2d pt2, Mat &c0, Mat &c1, Mat &c2, Mat &c3, Mat &c4)
	{
		Mat pt3d = app->LinearLSTriangulation(Point3d(pt1.x, pt1.y, 1), P1, Point3d(pt2.x, pt2.y, 1), P2);
		pt3d = (Mat_<double>(4, 1, CV_64F) << pt3d.at<double>(0), pt3d.at<double>(1), pt3d.at<double>(2), 1);

		double s1 = pt3d.dot(P1.row(2).t());
		double s2 = pt3d.dot(P2.row(2).t());
		s1s2 = s1 / s2;

		double dxx1 = (P1.at<double>(0, 0) - P1.at<double>(2, 0) * pt1.x);
		double dxy1 = (P1.at<double>(0, 1) - P1.at<double>(2, 1) * pt1.x);
		double dxz1 = (P1.at<double>(0, 2) - P1.at<double>(2, 2) * pt1.x);
		double dyx1 = (P1.at<double>(1, 0) - P1.at<double>(2, 0) * pt1.y);
		double dyy1 = (P1.at<double>(1, 1) - P1.at<double>(2, 1) * pt1.y);
		double dyz1 = (P1.at<double>(1, 2) - P1.at<double>(2, 2) * pt1.y);

		double dxx2 = (P2.at<double>(0, 0) - P2.at<double>(2, 0) * pt2.x);
		double dxy2 = (P2.at<double>(0, 1) - P2.at<double>(2, 1) * pt2.x);
		double dxz2 = (P2.at<double>(0, 2) - P2.at<double>(2, 2) * pt2.x);
		double dyx2 = (P2.at<double>(1, 0) - P2.at<double>(2, 0) * pt2.y);
		double dyy2 = (P2.at<double>(1, 1) - P2.at<double>(2, 1) * pt2.y);
		double dyz2 = (P2.at<double>(1, 2) - P2.at<double>(2, 2) * pt2.y);

		Mat deltaX1 = (Mat_<double>(3, 1) << dxx1, dxy1, dxz1);
		Mat deltaY1 = (Mat_<double>(3, 1) << dyx1, dyy1, dyz1);
		Mat deltaX2 = (Mat_<double>(3, 1) << dxx2, dxy2, dxz2);
		Mat deltaY2 = (Mat_<double>(3, 1) << dyx2, dyy2, dyz2);

		c0 = deltaX1.cross(deltaY1);
		c1 = deltaY1.cross(deltaX2);
		c2 = deltaX2.cross(deltaX1);
		c3 = deltaY1.cross(deltaY2);
		c4 = deltaY2.cross(deltaX1);
	}

protected:

	static double pso_evaluate(const std::vector<double> &vec, int dim, void *params) {
		return reinterpret_cast<objective_function*>(params)->evaluate(&vec[0], NULL, dim, 0);
	}

	static T bfgs_evaluate( void *instance, const T *x, T *g, const int n, const T step ) {
        return reinterpret_cast<objective_function*>(instance)->evaluate(x, g, n, step);
    }

    static int bfgs_progress( void *instance, const T *x, const T *g, const T fx, const T xnorm, const T gnorm, const T step, int n, int k, int ls ) {
        return reinterpret_cast<objective_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    T evaluate( const T *x, T *g, const int n, const T step )
    {
		Mat normal = FromSphericalNormal(x[0], x[1]);
		double transOnEpiline = x[2];

		Point2d pt2t = pt2;
		pt2t.x += epiline2.at<double>(0) * transOnEpiline;
		pt2t.y += epiline2.at<double>(1) * transOnEpiline;

		Mat c0, c1, c2, c3, c4;
		CalculateCrossVectors(pt1, pt2t, c0, c1, c2, c3, c4);
		
		double c = -normal.dot(c0);
		double mult = s1s2 / c;
		double a11 = normal.dot(c1) * mult;
		double a12 = normal.dot(c2) * mult;
		double a21 = normal.dot(c3) * mult;
		double a22 = normal.dot(c4) * mult;
		Mat HA = (Mat_<double>(3,3,CV_64F) << a11, a12, 0, a21, a22, 0, 0, 0, 1);

		if (determinant(HA) <= 0) // bikóz det(HA) = 1.0/det(HA)
			return 0;
		
		Mat A1 = Centering1 * HA * invCentering1;
		A1 = A1.rowRange(0,2);
		
		Mat A2;
		invertAffineTransform(A1, A2);
		
		Mat warpedPatch1;
		Mat warpedPatch2;
		warpAffine(*origPatch1, warpedPatch1, A1, (*origPatch1).size(), INTER_AREA, BORDER_TRANSPARENT);
		warpAffine(*origPatch2, warpedPatch2, A2, (*origPatch2).size(), INTER_AREA, BORDER_TRANSPARENT);

		double maxVal1 = abs(wzncc<double, 3>(warpedPatch1,*origPatch2,weights));
		double maxVal2 = abs(wzncc<double, 3>(warpedPatch2,*origPatch1,weights));
		//double maxVal1 = abs(zncc<double, 3>(warpedPatch1,*origPatch2));
		//double maxVal2 = abs(zncc<double, 3>(warpedPatch2,*origPatch1));
		return abs(maxVal1) * abs(maxVal2);
    }

    int progress( const T *x, const T *g, const T fx, const T xnorm, const T gnorm, const T step, int n, int k, int ls )
    {
        printf("Iteration %d:\n", k);
        printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
        printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
        printf("\n");
        return 0;
    }
};