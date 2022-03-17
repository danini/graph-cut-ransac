#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include "colmap_models.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace pose_lib {

struct CameraPose 
{
    // Rotation is represented as a unit quaternion
    // with real part first, i.e. QW, QX, QY, QZ
    Eigen::Vector4d q;
    Eigen::Vector3d t;

    inline Eigen::Vector3d quat_rotate(const Eigen::Vector4d &q, const Eigen::Vector3d &p) const
    {
        const double q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);
        const double p1 = p(0), p2 = p(1), p3 = p(2);
        const double px1 = -p1 * q2 - p2 * q3 - p3 * q4;
        const double px2 = p1 * q1 - p2 * q4 + p3 * q3;
        const double px3 = p2 * q1 + p1 * q4 - p3 * q2;
        const double px4 = p2 * q2 - p1 * q3 + p3 * q1;
        return Eigen::Vector3d(px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3, px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
                            px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1);
    }

    inline Eigen::Matrix3d quat_to_rotmat(const Eigen::Vector4d &q_) const {
        return Eigen::Quaterniond(q_(0), q_(1), q_(2), q_(3)).toRotationMatrix();
    }

    inline Eigen::Vector4d quat_conj(const Eigen::Vector4d &q) const { return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3)); }

    inline Eigen::Vector4d rotmat_to_quat(const Eigen::Matrix3d &R) const {
        Eigen::Quaterniond q_flip(R);
        Eigen::Vector4d q;
        q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
        q.normalize();
        return q;
    }

    inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d &qa, const Eigen::Vector4d &qb) const {
        const double qa1 = qa(0), qa2 = qa(1), qa3 = qa(2), qa4 = qa(3);
        const double qb1 = qb(0), qb2 = qb(1), qb3 = qb(2), qb4 = qb(3);

        return Eigen::Vector4d(qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4, qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
                            qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
                            qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1);
    }

    inline Eigen::Vector4d quat_exp(const Eigen::Vector3d &w) const {
        const double theta2 = w.squaredNorm();
        const double theta = std::sqrt(theta2);
        const double theta_half = 0.5 * theta;

        double re, im;
        if (theta > 1e-6) {
            re = std::cos(theta_half);
            im = std::sin(theta_half) / theta;
        } else {
            // we are close to zero, use taylor expansion to avoid problems
            // with zero divisors in sin(theta/2)/theta
            const double theta4 = theta2 * theta2;
            re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
            im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

            // for the linearized part we re-normalize to ensure unit length
            // here s should be roughly 1.0 anyways, so no problem with zero div
            const double s = std::sqrt(re * re + im * im * theta2);
            re /= s;
            im /= s;
        }
        return Eigen::Vector4d(re, im * w(0), im * w(1), im * w(2));
    }

    inline Eigen::Vector4d quat_step_pre(const Eigen::Vector4d &q, const Eigen::Vector3d &w_delta) const {
        return quat_multiply(quat_exp(w_delta), q);
    }
    inline Eigen::Vector4d quat_step_post(const Eigen::Vector4d &q, const Eigen::Vector3d &w_delta) const {
        return quat_multiply(q, quat_exp(w_delta));
    }

    // Constructors (Defaults to identity camera)
    CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
    CameraPose(const Eigen::Vector4d &qq, const Eigen::Vector3d &tt) : q(qq), t(tt) {}
    CameraPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &tt) : q(rotmat_to_quat(R)), t(tt) {}

    // Helper functions
    inline Eigen::Matrix3d R() const { return quat_to_rotmat(q); }
    inline Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Vector3d rotate(const Eigen::Vector3d &p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3d derotate(const Eigen::Vector3d &p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3d apply(const Eigen::Vector3d &p) const { return rotate(p) + t; }

    inline Eigen::Vector3d center() const { return -derotate(t); }
};

typedef std::vector<CameraPose> CameraPoseVector;

struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-8;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
};

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_pnp(const cv::Mat &correspondences_,
                    const size_t *sample_,
                    const size_t &sample_size_,
                    CameraPose *pose,
                    const BundleOptions &opt = BundleOptions(),
                    const double *weights = nullptr);

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int refine_pnp(const cv::Mat &correspondences_,
                    const size_t *sample_,
                    const size_t &sample_size_,
                    const Camera &camera,
                    CameraPose *pose,
                    const BundleOptions &opt = BundleOptions(),
                    const double *weights = nullptr);

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const cv::Mat &correspondences_,
                   const size_t *sample_,
                   const size_t &sample_size_,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions(),
                   const double *weights = nullptr);

// Fundamental matrix refinement. Minimizes Sampson error error.
// Returns number of iterations.
int refine_fundamental(const cv::Mat &correspondences_,
                       const size_t *sample_,
                       const size_t &sample_size_,
                       Eigen::Matrix3d *pose,
                       const BundleOptions &opt = BundleOptions(),
                       const double *weights = nullptr);

// Homography matrix refinement.
int refine_homography(const cv::Mat &correspondences_,
                      const size_t *sample_,
                      const size_t &sample_size_, 
                      Eigen::Matrix3d *H,
                      const BundleOptions &opt = BundleOptions(),
                      const double *weights = nullptr);

} // namespace pose_lib

#endif