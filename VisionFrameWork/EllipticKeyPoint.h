#pragma once

#include "stdafx.h"

using namespace cv;

class EllipticKeyPoint : public KeyPoint {
public:
    EllipticKeyPoint();
	EllipticKeyPoint(const EllipticKeyPoint& kp);
	EllipticKeyPoint(const KeyPoint& kp, const Mat_<double> Ai);
	virtual ~EllipticKeyPoint();

    static void convert( const std::vector<KeyPoint>& src, std::vector<EllipticKeyPoint>& dst );
    static void convert( const std::vector<EllipticKeyPoint>& src, std::vector<KeyPoint>& dst );

	Point2d applyAffineHomography(const Mat_<double>& H, const Point2d& pt);

	/*Size_<float> getAxes() const;
	Scalar getEllipse() const;*/

    Mat_<double> transformation;
};
