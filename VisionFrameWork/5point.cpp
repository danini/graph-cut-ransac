#include "5point.h"
#include "Polynomial.h"
#include "Rpoly.h"

using namespace std;
using namespace cv;

static void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4);
static cv::Mat TriangulatePoint(const cv::Point2d &pt1, const cv::Point2d &pt2, const cv::Mat &P1, const cv::Mat &P2);
static double CalcDepth(const cv::Mat &X, const cv::Mat &P);

bool Solve5PointEssential(std::vector<cv::Point2d> &pts1, std::vector<cv::Point2d> &pts2, cv::Mat &ret_E, cv::Mat &ret_P)
{
	int num_pts = pts1.size();

    assert(num_pts >= 5);
	
    if(num_pts < 5) {
        return false;
    }

    // F is a temp variable, not the F fundamental matrix
    cv::Mat F(num_pts, 9, CV_64F);

    for(int i=0; i < num_pts; i++) {
        float x1 = pts1[i].x;
		float y1 = pts1[i].y;

		float x2 = pts2[i].x;
		float y2 = pts2[i].y;

        F.at<double>(i,0) = x1*x2;
        F.at<double>(i,1) = x2*y1;
        F.at<double>(i,2) = x2;
        F.at<double>(i,3) = x1*y2;
        F.at<double>(i,4) = y1*y2;
        F.at<double>(i,5) = y2;
        F.at<double>(i,6) = x1;
        F.at<double>(i,7) = y1;
        F.at<double>(i,8) = 1.0;
    }

    cv::SVD svd(F, cv::SVD::FULL_UV);

    double e00 = svd.vt.at<double>(5,0);
    double e01 = svd.vt.at<double>(5,1);
    double e02 = svd.vt.at<double>(5,2);
    double e03 = svd.vt.at<double>(5,3);
    double e04 = svd.vt.at<double>(5,4);
    double e05 = svd.vt.at<double>(5,5);
    double e06 = svd.vt.at<double>(5,6);
    double e07 = svd.vt.at<double>(5,7);
    double e08 = svd.vt.at<double>(5,8);

    double e10 = svd.vt.at<double>(6,0);
    double e11 = svd.vt.at<double>(6,1);
    double e12 = svd.vt.at<double>(6,2);
    double e13 = svd.vt.at<double>(6,3);
    double e14 = svd.vt.at<double>(6,4);
    double e15 = svd.vt.at<double>(6,5);
    double e16 = svd.vt.at<double>(6,6);
    double e17 = svd.vt.at<double>(6,7);
    double e18 = svd.vt.at<double>(6,8);

    double e20 = svd.vt.at<double>(7,0);
    double e21 = svd.vt.at<double>(7,1);
    double e22 = svd.vt.at<double>(7,2);
    double e23 = svd.vt.at<double>(7,3);
    double e24 = svd.vt.at<double>(7,4);
    double e25 = svd.vt.at<double>(7,5);
    double e26 = svd.vt.at<double>(7,6);
    double e27 = svd.vt.at<double>(7,7);
    double e28 = svd.vt.at<double>(7,8);

    double e30 = svd.vt.at<double>(8,0);
    double e31 = svd.vt.at<double>(8,1);
    double e32 = svd.vt.at<double>(8,2);
    double e33 = svd.vt.at<double>(8,3);
    double e34 = svd.vt.at<double>(8,4);
    double e35 = svd.vt.at<double>(8,5);
    double e36 = svd.vt.at<double>(8,6);
    double e37 = svd.vt.at<double>(8,7);
    double e38 = svd.vt.at<double>(8,8);

    // Out symbolic polynomial matrix
    PolyMatrix M(10,10);

    // This file is not pretty to look at ...
    #include "Mblock.h"

    // symbolic determinant using interpolation based on the papers:
    // "Symbolic Determinants: Calculating the Degree", http://www.cs.tamu.edu/academics/tr/tamu-cs-tr-2005-7-1
    // "Multivariate Determinants Through Univariate Interpolation", http://www.cs.tamu.edu/academics/tr/tamu-cs-tr-2005-7-2

    // max power of the determinant is x^10, so we need 11 points for interpolation
    // the 11 points are at x = [-5, -4 .... 4, 5], luckily there is no overflow at x^10

    cv::Mat X = cv::Mat::ones(11, 11, CV_64F);
    cv::Mat b(11, 1, CV_64F);
    cv::Mat ret_eval(10, 10, CV_64F);

    // first column of M is the lowest power
    for(int i=-5, j=0; i <= 5; i++, j++) {
        M.Eval(i, (double*)ret_eval.data);

        double t = i;

        for(int k=1; k < 11; k++) {
            X.at<double>(j,k) = t;
            t *= i;
        }

        b.at<double>(j,0) = cv::determinant(ret_eval);
    }

    cv::Mat a = X.inv()*b;

    // Solve for z
    int degrees = 10;
    double coeffs[11];
    double zeror[11], zeroi[11];
    vector <double> solutions;

    // rpoly_ak1 expects highest power first
    for(int i=0; i < a.rows; i++) {
        coeffs[i] = a.at<double>(a.rows-i-1);
    }

    // Find roots of polynomial
    rpoly_ak1(coeffs, &degrees, zeror, zeroi);

    for(int i=0; i < degrees; i++) {
        if(zeroi[i] == 0) {
            solutions.push_back(zeror[i]);
        }
    }

    //cout << "Found " << solutions.size() << " solutions!" << endl;

    if(solutions.empty()) {
        return false;
    }

    // Back substitute the z values and compute null space to get x,y
    cv::Mat x1(3, 1, CV_64F);
    cv::Mat x2(3, 1, CV_64F);
    cv::Mat E = cv::Mat::zeros(3,3,CV_64F);
    cv::Mat P_ref = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P[4];
    cv::Mat pt3d;

	std::vector<cv::Mat> Es, Ps;
	std::vector<int> inliers;
	int bestInlierIdx;
	int bestInlierNumber = 0;

    for(size_t i=0; i < solutions.size(); i++) {
        double z = solutions[i];

        M.Eval(z, (double*)ret_eval.data);

        // re-use the svd variable
        svd(ret_eval);

        // svd.vt.row(9) represents
        // [x^3 , y^3 , x^2 y, xy^2 , x^2 , y^2 , xy, x, y, 1]^T

        // Scale it so the last element is 1, to get the correct answer
        double x = svd.vt.at<double>(9,7) / svd.vt.at<double>(9,9);;
        double y = svd.vt.at<double>(9,8) / svd.vt.at<double>(9,9);;

        // Build the essential matrix from all the known x,y,z values
        E.at<double>(0,0) = e00*x + e10*y + e20*z + e30;
        E.at<double>(0,1) = e01*x + e11*y + e21*z + e31;
        E.at<double>(0,2) = e02*x + e12*y + e22*z + e32;

        E.at<double>(1,0) = e03*x + e13*y + e23*z + e33;
        E.at<double>(1,1) = e04*x + e14*y + e24*z + e34;
        E.at<double>(1,2) = e05*x + e15*y + e25*z + e35;

        E.at<double>(2,0) = e06*x + e16*y + e26*z + e36;
        E.at<double>(2,1) = e07*x + e17*y + e27*z + e37;
        E.at<double>(2,2) = e08*x + e18*y + e28*z + e38;

        x1.at<double>(0,0) = pts1[0].x;
        x1.at<double>(1,0) = pts1[0].y;
        x1.at<double>(2,0) = 1.0;

        x2.at<double>(0,0) = pts2[0].x;
        x2.at<double>(1,0) = pts2[0].y;
        x2.at<double>(2,0) = 1.0;

        // Test to see if this E matrix is the correct one we're after
        ProjectionsFromEssential(E, P[0], P[1], P[2], P[3]);

        cv::Mat best_E, best_P;
        int best_inliers = 0;
        bool found = false;

        for(int j=0; j < 4; j++) {
            pt3d = TriangulatePoint(pts1[0], pts2[0], P_ref, P[j]);
            double depth1 = CalcDepth(pt3d, P_ref);
            double depth2 = CalcDepth(pt3d, P[j]);

            if(depth1 > 0 && depth2 > 0){
                int inliers = 1; // number of points in front of the camera

                for(int k=1; k < num_pts; k++) {
                    pt3d = TriangulatePoint(pts1[k], pts2[k], P_ref, P[j]);
                    depth1 = CalcDepth(pt3d, P_ref);
                    depth2 = CalcDepth(pt3d, P[j]);

                    if(depth1 > 0 && depth2 > 0) {
                        inliers++;
                    }
                }

                if(inliers > best_inliers && inliers >= 5) {
                    best_inliers = inliers;

                    E.copyTo(best_E);
                    P[j].copyTo(best_P);
                    found = true;
                }

                // Special case, with 5 points you can get a perfect solution
                if(num_pts == 5 && inliers == 5) {
                    break;
                }
            }
        }

        if(found) {
            Es.push_back(best_E);
            Ps.push_back(best_P);
            inliers.push_back(best_inliers);

			if (best_inliers > bestInlierNumber)
			{
				bestInlierIdx = Es.size() - 1;
				bestInlierNumber = best_inliers;
			}
        }
    }

    if(!Es.size()) {
        return false;
    }

	ret_E = Es[bestInlierIdx];
	ret_P = Ps[bestInlierIdx];
    return true;
}

// X is 4x1 is [x,y,z,w]
// P is 3x4 projection matrix
double CalcDepth(const cv::Mat &X, const cv::Mat &P)
{
    // back project
    cv::Mat X2 = P*X;

    double det = cv::determinant(P(cv::Range(0,3), cv::Range(0,3)));
    double w = X2.at<double>(2,0);
    double W = X.at<double>(3,0);

    double a = P.at<double>(0,2);
    double b = P.at<double>(1,2);
    double c = P.at<double>(2,2);

    double m3 = sqrt(a*a + b*b + c*c);  // 3rd column of M

    double sign;

    if(det > 0) {
        sign = 1;
    }
    else {
        sign = -1;
    }

    return (w/W)*(sign/m3);
}

cv::Mat TriangulatePoint(const cv::Point2d &pt1, const cv::Point2d &pt2, const cv::Mat &P1, const cv::Mat &P2)
{
    cv::Mat A(4,4,CV_64F);

	A.at<double>(0,0) = pt1.x*P1.at<double>(2,0) - P1.at<double>(0,0);
	A.at<double>(0,1) = pt1.x*P1.at<double>(2,1) - P1.at<double>(0,1);
	A.at<double>(0,2) = pt1.x*P1.at<double>(2,2) - P1.at<double>(0,2);
	A.at<double>(0,3) = pt1.x*P1.at<double>(2,3) - P1.at<double>(0,3);

	A.at<double>(1,0) = pt1.y*P1.at<double>(2,0) - P1.at<double>(1,0);
	A.at<double>(1,1) = pt1.y*P1.at<double>(2,1) - P1.at<double>(1,1);
	A.at<double>(1,2) = pt1.y*P1.at<double>(2,2) - P1.at<double>(1,2);
	A.at<double>(1,3) = pt1.y*P1.at<double>(2,3) - P1.at<double>(1,3);

	A.at<double>(2,0) = pt2.x*P2.at<double>(2,0) - P2.at<double>(0,0);
	A.at<double>(2,1) = pt2.x*P2.at<double>(2,1) - P2.at<double>(0,1);
	A.at<double>(2,2) = pt2.x*P2.at<double>(2,2) - P2.at<double>(0,2);
	A.at<double>(2,3) = pt2.x*P2.at<double>(2,3) - P2.at<double>(0,3);

	A.at<double>(3,0) = pt2.y*P2.at<double>(2,0) - P2.at<double>(1,0);
	A.at<double>(3,1) = pt2.y*P2.at<double>(2,1) - P2.at<double>(1,1);
	A.at<double>(3,2) = pt2.y*P2.at<double>(2,2) - P2.at<double>(1,2);
	A.at<double>(3,3) = pt2.y*P2.at<double>(2,3) - P2.at<double>(1,3);

	cv::SVD svd(A);

    cv::Mat X(4,1,CV_64F);

	X.at<double>(0,0) = svd.vt.at<double>(3,0);
	X.at<double>(1,0) = svd.vt.at<double>(3,1);
	X.at<double>(2,0) = svd.vt.at<double>(3,2);
	X.at<double>(3,0) = svd.vt.at<double>(3,3);

    return X;
}

void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4)
{
    // Assumes input E is a rank 2 matrix, with equal singular values
    cv::SVD svd(E);

    cv::Mat VT = svd.vt;
    cv::Mat V = svd.vt.t();
    cv::Mat U = svd.u;
    cv::Mat W = cv::Mat::zeros(3,3,CV_64F);

	// Find rotation, translation
	W.at<double>(0,1) = -1.0;
	W.at<double>(1,0) = 1.0;
	W.at<double>(2,2) = 1.0;

	// P1, P2, P3, P4
	P1 = cv::Mat::eye(3,4,CV_64F);
	P2 = cv::Mat::eye(3,4,CV_64F);
	P3 = cv::Mat::eye(3,4,CV_64F);
	P4 = cv::Mat::eye(3,4,CV_64F);

    // Rotation
	P1(cv::Range(0,3), cv::Range(0,3)) = U*W*VT;
	P2(cv::Range(0,3), cv::Range(0,3)) = U*W*VT;
	P3(cv::Range(0,3), cv::Range(0,3)) = U*W.t()*VT;
	P4(cv::Range(0,3), cv::Range(0,3)) = U*W.t()*VT;

    // Translation
    P1(cv::Range::all(), cv::Range(3,4)) = U(cv::Range::all(), cv::Range(2,3))*1;
    P2(cv::Range::all(), cv::Range(3,4)) = -U(cv::Range::all(), cv::Range(2,3));
    P3(cv::Range::all(), cv::Range(3,4)) = U(cv::Range::all(), cv::Range(2,3))*1;
    P4(cv::Range::all(), cv::Range(3,4)) = -U(cv::Range::all(), cv::Range(2,3));
}

