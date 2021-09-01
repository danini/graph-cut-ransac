#include "bundle.h"
#include "jacobian_impl.h"
#include "robust_loss.h"
#include <opencv2/core.hpp>

namespace pose_lib {

#define SWITCH_LOSS_FUNCTIONS                     \
    case BundleOptions::LossType::TRIVIAL:        \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);   \
        break;                                    \
    case BundleOptions::LossType::TRUNCATED:      \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss); \
        break;                                    \
    case BundleOptions::LossType::HUBER:          \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);     \
        break;                                    \
    case BundleOptions::LossType::CAUCHY:         \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);    \
        break;

template <typename JacobianAccumulator>
int lm_6dof_impl(const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;
        JtJ(5, 5) += lambda;

        Eigen::Matrix<double, 6, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        pose_new.R = pose->R + pose->R * (a * sw + (1 - b) * sw * sw);
        pose_new.t = pose->t + pose->R * sol.block<3, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            JtJ(5, 5) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename JacobianAccumulator>
int lm_5dof_impl(const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 5, 5> JtJ;
    Eigen::Matrix<double, 5, 1> Jtr;
    Eigen::Matrix<double, 3, 2> tangent_basis;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr, tangent_basis);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;

        Eigen::Matrix<double, 5, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        pose_new.R = pose->R + pose->R * (a * sw + (1 - b) * sw * sw);
        // In contrast to the 6dof case, we don't apply R here
        // (since this can already be added into tangent_basis)
        pose_new.t = pose->t + tangent_basis * sol.block<2, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, CameraPose *pose, const BundleOptions &opt) {
    pose_lib::Camera camera;
    camera.model_id = -1;
    return bundle_adjust(x, X, camera, pose, opt);
}

// helper function to dispatch to the correct camera model (we do it once here to avoid doing it in every iteration)
template <typename LossFunction>
int dispatch_bundle_camera_model(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const LossFunction &loss) {
    switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model) \
    case Model::model_id:               \
        return lm_6dof_impl<CameraJacobianAccumulator<Model, decltype(loss)>>(CameraJacobianAccumulator<Model, decltype(loss)>(x, X, camera, loss), pose, opt);

        SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
    }
    return -1;
}

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt) {
    // TODO try rescaling image camera.rescale(1.0 / camera.focal()) and image points
    switch (opt.loss_type) {
    case BundleOptions::LossType::TRIVIAL: {
        TrivialLoss loss_fn;
        return dispatch_bundle_camera_model<TrivialLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::TRUNCATED: {
        TruncatedLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<TruncatedLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::HUBER: {
        HuberLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<HuberLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::CAUCHY: {
        CauchyLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<CauchyLoss>(x, X, camera, pose, opt, loss_fn);
    }
    default:
        return -1;
    }
}

int refine_relpose(const cv::Mat &correspondences_,
                    const size_t *sample_,
                    const size_t &sample_size_,
                    CameraPose *pose, 
                    const BundleOptions &opt,
                    const std::vector<double> &weights) 
{
    if (weights.size() != sample_size_) 
    {
        // We have per-residual weights
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                             \
        {                                                                                                       \
            LossFunction loss_fn(opt.loss_scale);                                                               \
            RelativePoseJacobianAccumulator<LossFunction, std::vector<double>> accum(correspondences_, sample_, sample_size_, loss_fn, weights); \
            return lm_5dof_impl<decltype(accum)>(accum, pose, opt);                                             \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            return -1;
        };
    } else {
        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                               \
        {                                                                         \
            LossFunction loss_fn(opt.loss_scale);                                 \
            RelativePoseJacobianAccumulator<LossFunction> accum(correspondences_, sample_, sample_size_, loss_fn); \
            return lm_5dof_impl<decltype(accum)>(accum, pose, opt);               \
        }
            SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

} // namespace pose_lib