// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "solver_engine.h"
#include "fundamental_estimator.h"
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class P1ACQuerySolver : public SolverEngine
			{
			public:
				P1ACQuerySolver()
				{
				}

				~P1ACQuerySolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 8;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 1;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				OLGA_INLINE Eigen::MatrixXcd solver_autogen(const Eigen::VectorXd& data) const;
				OLGA_INLINE void unpack_solution( const Eigen::VectorXd &soln, Eigen::Matrix3d &R, Eigen::Vector3d &t ) const;
			};
			
			OLGA_INLINE bool P1ACQuerySolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				const double * data_ptr = reinterpret_cast<double *>(data_.row(sample_[0]).data);
				const size_t columns = data_.cols;

				const double &px = data_ptr[0], // Coordinate X in the query image
					&py = data_ptr[1], // Coordinate Y in the query image
					&qz = data_ptr[4], // Coordinate Z in the reference image
					&nx = data_ptr[5], // Coordinate X of the normal in the reference image
					&ny = data_ptr[6], // Coordinate Y of the normal in the reference image
					&nz = data_ptr[7], // Coordinate Z of the normal in the reference image
					&a1 = data_ptr[8], // Element a11 of the affine transformation
					&a2 = data_ptr[9], // Element a12 of the affine transformation
					&a3 = data_ptr[10], // Element a21 of the affine transformation
					&a4 = data_ptr[11]; // Element a22 of the affine transformation

				const double
					qx = data_ptr[2] / qz, // Coordinate X in the reference image
					qy = data_ptr[3] / qz; // Coordinate Y in the reference image

				// The next 3*3 elements are the rotation matrix of the reference frame in a row-major order
				Eigen::Matrix3d R_ref;
				R_ref << data_ptr[12], data_ptr[13], data_ptr[14],
					data_ptr[15], data_ptr[16], data_ptr[17],
					data_ptr[18], data_ptr[19], data_ptr[20];

				// The next 3 elements are the translation vector of the reference frame
				Eigen::Vector3d t_ref;
				t_ref << data_ptr[21], data_ptr[22], data_ptr[23];

				// Build linear constraints
				Eigen::MatrixXd M(6,12);
				M << -qz*qx, 0, qz*qx*px, -qz*qy, 0, qz*qy*px, -qz, 0, qz*px, -1, 0, px,
					0, -qz*qx, qz*qx*py, 0, -qz*qy, qz*qy*py, 0, -qz, qz*py, 0, -1, py,
					- qz*nz - qz*nx*qx - qz*ny*qy, 0, qz*nz*px + qz*nx*qx*px + qz*ny*qy*px + a1*qz*nx*qx*qx + a1*qz*nz*qx + a1*qz*ny*qx*qy, 0, 0, a1*qz*ny*qy*qy + a1*qz*nz*qy + a1*qz*nx*qx*qy, 0, 0, a1*qz*nz + a1*qz*nx*qx + a1*qz*ny*qy, -nx, 0, a1*nz + nx*px + a1*nx*qx + a1*ny*qy,
					0, - qz*nz - qz*nx*qx - qz*ny*qy, qz*nz*py + qz*nx*qx*py + qz*ny*qy*py + a3*qz*nx*qx*qx + a3*qz*nz*qx + a3*qz*ny*qx*qy, 0, 0, a3*qz*ny*qy*qy + a3*qz*nz*qy + a3*qz*nx*qx*qy, 0, 0, a3*qz*nz + a3*qz*nx*qx + a3*qz*ny*qy, 0, -nx, a3*nz + nx*py + a3*nx*qx + a3*ny*qy,
					0, 0, a2*qz*nx*qx*qx + a2*qz*nz*qx + a2*qz*ny*qx*qy, - qz*nz - qz*nx*qx - qz*ny*qy, 0, qz*nz*px + qz*nx*qx*px + qz*ny*qy*px + a2*qz*ny*qy*qy + a2*qz*nz*qy + a2*qz*nx*qx*qy, 0, 0, a2*qz*nz + a2*qz*nx*qx + a2*qz*ny*qy, -ny, 0, a2*nz + ny*px + a2*nx*qx + a2*ny*qy,
					0, 0, a4*qz*nx*qx*qx + a4*qz*nz*qx + a4*qz*ny*qx*qy, 0, - qz*nz - qz*nx*qx - qz*ny*qy, qz*nz*py + qz*nx*qx*py + qz*ny*qy*py + a4*qz*ny*qy*qy + a4*qz*nz*qy + a4*qz*nx*qx*qy, 0, 0, a4*qz*nz + a4*qz*nx*qx + a4*qz*ny*qy, 0, -ny, a4*nz + ny*py + a4*nx*qx + a4*ny*qy;
				
				// get nullspace of M
				Eigen::MatrixXd V = M.jacobiSvd(Eigen::ComputeFullV).matrixV();
				Eigen::MatrixXd Bmat = V.block(0,6,12,6);
				Eigen::VectorXd Bvec(12*6);
				for ( size_t i = 0; i < 6; i++ ) 
					Bvec.block(12*i,0,12,1) = V.col(i+6);
				
				// Solve for linear coefficients of null vectors
				Eigen::MatrixXcd bsolns = solver_autogen(Bvec);

				int nsolns = bsolns.cols();
				Eigen::MatrixXd solns(12,nsolns);
				
				Eigen::Matrix3d R;
				Eigen::Vector3d t;

				for ( int i = 0; i < nsolns; i++ )
				{
					// extract solution for linear coefficients of null vectors
					Eigen::VectorXd b(6);
					b.head(5) = bsolns.col(i).real();
					b(5) = 1;
					
					// make linear combination of null vectors
					solns.col(i) = Bmat*b;
					//  R1_1, R2_1, R3_1, R1_2, R2_2, R3_2, R1_3, R2_3, R3_3, t1, t2, t3

					unpack_solution(solns.col(i), R, t);
					
					Eigen::Matrix3d R_world = R * R_ref;
					Eigen::Vector3d t_world = R * t_ref + t;

					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << R_world, t_world;
					models_.emplace_back(model);
				}
				
				return models_.size();
			}
			
			OLGA_INLINE void P1ACQuerySolver::unpack_solution( const Eigen::VectorXd &soln, Eigen::Matrix3d &R, Eigen::Vector3d &t ) const
			{
				R.col(0) = soln.block(0,0,3,1);
				R.col(1) = soln.block(3,0,3,1);
				R.col(2) = soln.block(6,0,3,1);
				t = soln.block(9,0,3,1);

				// correct scale
				double s = R.col(0).norm();
				R /= s;
				t /= s;
				if ( R.determinant() < 0 )
				{
					R = -R;
					t = -t;
				}
			}
			
			OLGA_INLINE Eigen::MatrixXcd P1ACQuerySolver::solver_autogen(const Eigen::VectorXd& data) const
			{
				// Compute coefficients
				const double* d = data.data();
				Eigen::VectorXd coeffs(210);
				coeffs[0] = std::pow(d[0],2) + std::pow(d[1],2) + std::pow(d[2],2) - std::pow(d[3],2) - std::pow(d[4],2) - std::pow(d[5],2);
				coeffs[1] = 2*d[0]*d[12] + 2*d[1]*d[13] + 2*d[2]*d[14] - 2*d[3]*d[15] - 2*d[4]*d[16] - 2*d[5]*d[17];
				coeffs[2] = std::pow(d[12],2) + std::pow(d[13],2) + std::pow(d[14],2) - std::pow(d[15],2) - std::pow(d[16],2) - std::pow(d[17],2);
				coeffs[3] = 2*d[0]*d[24] + 2*d[1]*d[25] + 2*d[2]*d[26] - 2*d[3]*d[27] - 2*d[4]*d[28] - 2*d[5]*d[29];
				coeffs[4] = 2*d[12]*d[24] + 2*d[13]*d[25] + 2*d[14]*d[26] - 2*d[15]*d[27] - 2*d[16]*d[28] - 2*d[17]*d[29];
				coeffs[5] = std::pow(d[24],2) + std::pow(d[25],2) + std::pow(d[26],2) - std::pow(d[27],2) - std::pow(d[28],2) - std::pow(d[29],2);
				coeffs[6] = 2*d[0]*d[36] + 2*d[1]*d[37] + 2*d[2]*d[38] - 2*d[3]*d[39] - 2*d[4]*d[40] - 2*d[5]*d[41];
				coeffs[7] = 2*d[12]*d[36] + 2*d[13]*d[37] + 2*d[14]*d[38] - 2*d[15]*d[39] - 2*d[16]*d[40] - 2*d[17]*d[41];
				coeffs[8] = 2*d[24]*d[36] + 2*d[25]*d[37] + 2*d[26]*d[38] - 2*d[27]*d[39] - 2*d[28]*d[40] - 2*d[29]*d[41];
				coeffs[9] = std::pow(d[36],2) + std::pow(d[37],2) + std::pow(d[38],2) - std::pow(d[39],2) - std::pow(d[40],2) - std::pow(d[41],2);
				coeffs[10] = 2*d[0]*d[48] + 2*d[1]*d[49] + 2*d[2]*d[50] - 2*d[3]*d[51] - 2*d[4]*d[52] - 2*d[5]*d[53];
				coeffs[11] = 2*d[12]*d[48] + 2*d[13]*d[49] + 2*d[14]*d[50] - 2*d[15]*d[51] - 2*d[16]*d[52] - 2*d[17]*d[53];
				coeffs[12] = 2*d[24]*d[48] + 2*d[25]*d[49] + 2*d[26]*d[50] - 2*d[27]*d[51] - 2*d[28]*d[52] - 2*d[29]*d[53];
				coeffs[13] = 2*d[36]*d[48] + 2*d[37]*d[49] + 2*d[38]*d[50] - 2*d[39]*d[51] - 2*d[40]*d[52] - 2*d[41]*d[53];
				coeffs[14] = std::pow(d[48],2) + std::pow(d[49],2) + std::pow(d[50],2) - std::pow(d[51],2) - std::pow(d[52],2) - std::pow(d[53],2);
				coeffs[15] = 2*d[0]*d[60] + 2*d[1]*d[61] + 2*d[2]*d[62] - 2*d[3]*d[63] - 2*d[4]*d[64] - 2*d[5]*d[65];
				coeffs[16] = 2*d[12]*d[60] + 2*d[13]*d[61] + 2*d[14]*d[62] - 2*d[15]*d[63] - 2*d[16]*d[64] - 2*d[17]*d[65];
				coeffs[17] = 2*d[24]*d[60] + 2*d[25]*d[61] + 2*d[26]*d[62] - 2*d[27]*d[63] - 2*d[28]*d[64] - 2*d[29]*d[65];
				coeffs[18] = 2*d[36]*d[60] + 2*d[37]*d[61] + 2*d[38]*d[62] - 2*d[39]*d[63] - 2*d[40]*d[64] - 2*d[41]*d[65];
				coeffs[19] = 2*d[48]*d[60] + 2*d[49]*d[61] + 2*d[50]*d[62] - 2*d[51]*d[63] - 2*d[52]*d[64] - 2*d[53]*d[65];
				coeffs[20] = std::pow(d[60],2) + std::pow(d[61],2) + std::pow(d[62],2) - std::pow(d[63],2) - std::pow(d[64],2) - std::pow(d[65],2);
				coeffs[21] = std::pow(d[0],2) + std::pow(d[1],2) + std::pow(d[2],2) - std::pow(d[6],2) - std::pow(d[7],2) - std::pow(d[8],2);
				coeffs[22] = 2*d[0]*d[12] + 2*d[1]*d[13] + 2*d[2]*d[14] - 2*d[6]*d[18] - 2*d[7]*d[19] - 2*d[8]*d[20];
				coeffs[23] = std::pow(d[12],2) + std::pow(d[13],2) + std::pow(d[14],2) - std::pow(d[18],2) - std::pow(d[19],2) - std::pow(d[20],2);
				coeffs[24] = 2*d[0]*d[24] + 2*d[1]*d[25] + 2*d[2]*d[26] - 2*d[6]*d[30] - 2*d[7]*d[31] - 2*d[8]*d[32];
				coeffs[25] = 2*d[12]*d[24] + 2*d[13]*d[25] + 2*d[14]*d[26] - 2*d[18]*d[30] - 2*d[19]*d[31] - 2*d[20]*d[32];
				coeffs[26] = std::pow(d[24],2) + std::pow(d[25],2) + std::pow(d[26],2) - std::pow(d[30],2) - std::pow(d[31],2) - std::pow(d[32],2);
				coeffs[27] = 2*d[0]*d[36] + 2*d[1]*d[37] + 2*d[2]*d[38] - 2*d[6]*d[42] - 2*d[7]*d[43] - 2*d[8]*d[44];
				coeffs[28] = 2*d[12]*d[36] + 2*d[13]*d[37] + 2*d[14]*d[38] - 2*d[18]*d[42] - 2*d[19]*d[43] - 2*d[20]*d[44];
				coeffs[29] = 2*d[24]*d[36] + 2*d[25]*d[37] + 2*d[26]*d[38] - 2*d[30]*d[42] - 2*d[31]*d[43] - 2*d[32]*d[44];
				coeffs[30] = std::pow(d[36],2) + std::pow(d[37],2) + std::pow(d[38],2) - std::pow(d[42],2) - std::pow(d[43],2) - std::pow(d[44],2);
				coeffs[31] = 2*d[0]*d[48] + 2*d[1]*d[49] + 2*d[2]*d[50] - 2*d[6]*d[54] - 2*d[7]*d[55] - 2*d[8]*d[56];
				coeffs[32] = 2*d[12]*d[48] + 2*d[13]*d[49] + 2*d[14]*d[50] - 2*d[18]*d[54] - 2*d[19]*d[55] - 2*d[20]*d[56];
				coeffs[33] = 2*d[24]*d[48] + 2*d[25]*d[49] + 2*d[26]*d[50] - 2*d[30]*d[54] - 2*d[31]*d[55] - 2*d[32]*d[56];
				coeffs[34] = 2*d[36]*d[48] + 2*d[37]*d[49] + 2*d[38]*d[50] - 2*d[42]*d[54] - 2*d[43]*d[55] - 2*d[44]*d[56];
				coeffs[35] = std::pow(d[48],2) + std::pow(d[49],2) + std::pow(d[50],2) - std::pow(d[54],2) - std::pow(d[55],2) - std::pow(d[56],2);
				coeffs[36] = 2*d[0]*d[60] + 2*d[1]*d[61] + 2*d[2]*d[62] - 2*d[6]*d[66] - 2*d[7]*d[67] - 2*d[8]*d[68];
				coeffs[37] = 2*d[12]*d[60] + 2*d[13]*d[61] + 2*d[14]*d[62] - 2*d[18]*d[66] - 2*d[19]*d[67] - 2*d[20]*d[68];
				coeffs[38] = 2*d[24]*d[60] + 2*d[25]*d[61] + 2*d[26]*d[62] - 2*d[30]*d[66] - 2*d[31]*d[67] - 2*d[32]*d[68];
				coeffs[39] = 2*d[36]*d[60] + 2*d[37]*d[61] + 2*d[38]*d[62] - 2*d[42]*d[66] - 2*d[43]*d[67] - 2*d[44]*d[68];
				coeffs[40] = 2*d[48]*d[60] + 2*d[49]*d[61] + 2*d[50]*d[62] - 2*d[54]*d[66] - 2*d[55]*d[67] - 2*d[56]*d[68];
				coeffs[41] = std::pow(d[60],2) + std::pow(d[61],2) + std::pow(d[62],2) - std::pow(d[66],2) - std::pow(d[67],2) - std::pow(d[68],2);
				coeffs[42] = std::pow(d[0],2) - std::pow(d[1],2) + std::pow(d[3],2) - std::pow(d[4],2) + std::pow(d[6],2) - std::pow(d[7],2);
				coeffs[43] = 2*d[0]*d[12] - 2*d[1]*d[13] + 2*d[3]*d[15] - 2*d[4]*d[16] + 2*d[6]*d[18] - 2*d[7]*d[19];
				coeffs[44] = std::pow(d[12],2) - std::pow(d[13],2) + std::pow(d[15],2) - std::pow(d[16],2) + std::pow(d[18],2) - std::pow(d[19],2);
				coeffs[45] = 2*d[0]*d[24] - 2*d[1]*d[25] + 2*d[3]*d[27] - 2*d[4]*d[28] + 2*d[6]*d[30] - 2*d[7]*d[31];
				coeffs[46] = 2*d[12]*d[24] - 2*d[13]*d[25] + 2*d[15]*d[27] - 2*d[16]*d[28] + 2*d[18]*d[30] - 2*d[19]*d[31];
				coeffs[47] = std::pow(d[24],2) - std::pow(d[25],2) + std::pow(d[27],2) - std::pow(d[28],2) + std::pow(d[30],2) - std::pow(d[31],2);
				coeffs[48] = 2*d[0]*d[36] - 2*d[1]*d[37] + 2*d[3]*d[39] - 2*d[4]*d[40] + 2*d[6]*d[42] - 2*d[7]*d[43];
				coeffs[49] = 2*d[12]*d[36] - 2*d[13]*d[37] + 2*d[15]*d[39] - 2*d[16]*d[40] + 2*d[18]*d[42] - 2*d[19]*d[43];
				coeffs[50] = 2*d[24]*d[36] - 2*d[25]*d[37] + 2*d[27]*d[39] - 2*d[28]*d[40] + 2*d[30]*d[42] - 2*d[31]*d[43];
				coeffs[51] = std::pow(d[36],2) - std::pow(d[37],2) + std::pow(d[39],2) - std::pow(d[40],2) + std::pow(d[42],2) - std::pow(d[43],2);
				coeffs[52] = 2*d[0]*d[48] - 2*d[1]*d[49] + 2*d[3]*d[51] - 2*d[4]*d[52] + 2*d[6]*d[54] - 2*d[7]*d[55];
				coeffs[53] = 2*d[12]*d[48] - 2*d[13]*d[49] + 2*d[15]*d[51] - 2*d[16]*d[52] + 2*d[18]*d[54] - 2*d[19]*d[55];
				coeffs[54] = 2*d[24]*d[48] - 2*d[25]*d[49] + 2*d[27]*d[51] - 2*d[28]*d[52] + 2*d[30]*d[54] - 2*d[31]*d[55];
				coeffs[55] = 2*d[36]*d[48] - 2*d[37]*d[49] + 2*d[39]*d[51] - 2*d[40]*d[52] + 2*d[42]*d[54] - 2*d[43]*d[55];
				coeffs[56] = std::pow(d[48],2) - std::pow(d[49],2) + std::pow(d[51],2) - std::pow(d[52],2) + std::pow(d[54],2) - std::pow(d[55],2);
				coeffs[57] = 2*d[0]*d[60] - 2*d[1]*d[61] + 2*d[3]*d[63] - 2*d[4]*d[64] + 2*d[6]*d[66] - 2*d[7]*d[67];
				coeffs[58] = 2*d[12]*d[60] - 2*d[13]*d[61] + 2*d[15]*d[63] - 2*d[16]*d[64] + 2*d[18]*d[66] - 2*d[19]*d[67];
				coeffs[59] = 2*d[24]*d[60] - 2*d[25]*d[61] + 2*d[27]*d[63] - 2*d[28]*d[64] + 2*d[30]*d[66] - 2*d[31]*d[67];
				coeffs[60] = 2*d[36]*d[60] - 2*d[37]*d[61] + 2*d[39]*d[63] - 2*d[40]*d[64] + 2*d[42]*d[66] - 2*d[43]*d[67];
				coeffs[61] = 2*d[48]*d[60] - 2*d[49]*d[61] + 2*d[51]*d[63] - 2*d[52]*d[64] + 2*d[54]*d[66] - 2*d[55]*d[67];
				coeffs[62] = std::pow(d[60],2) - std::pow(d[61],2) + std::pow(d[63],2) - std::pow(d[64],2) + std::pow(d[66],2) - std::pow(d[67],2);
				coeffs[63] = std::pow(d[0],2) - std::pow(d[2],2) + std::pow(d[3],2) - std::pow(d[5],2) + std::pow(d[6],2) - std::pow(d[8],2);
				coeffs[64] = 2*d[0]*d[12] - 2*d[2]*d[14] + 2*d[3]*d[15] - 2*d[5]*d[17] + 2*d[6]*d[18] - 2*d[8]*d[20];
				coeffs[65] = std::pow(d[12],2) - std::pow(d[14],2) + std::pow(d[15],2) - std::pow(d[17],2) + std::pow(d[18],2) - std::pow(d[20],2);
				coeffs[66] = 2*d[0]*d[24] - 2*d[2]*d[26] + 2*d[3]*d[27] - 2*d[5]*d[29] + 2*d[6]*d[30] - 2*d[8]*d[32];
				coeffs[67] = 2*d[12]*d[24] - 2*d[14]*d[26] + 2*d[15]*d[27] - 2*d[17]*d[29] + 2*d[18]*d[30] - 2*d[20]*d[32];
				coeffs[68] = std::pow(d[24],2) - std::pow(d[26],2) + std::pow(d[27],2) - std::pow(d[29],2) + std::pow(d[30],2) - std::pow(d[32],2);
				coeffs[69] = 2*d[0]*d[36] - 2*d[2]*d[38] + 2*d[3]*d[39] - 2*d[5]*d[41] + 2*d[6]*d[42] - 2*d[8]*d[44];
				coeffs[70] = 2*d[12]*d[36] - 2*d[14]*d[38] + 2*d[15]*d[39] - 2*d[17]*d[41] + 2*d[18]*d[42] - 2*d[20]*d[44];
				coeffs[71] = 2*d[24]*d[36] - 2*d[26]*d[38] + 2*d[27]*d[39] - 2*d[29]*d[41] + 2*d[30]*d[42] - 2*d[32]*d[44];
				coeffs[72] = std::pow(d[36],2) - std::pow(d[38],2) + std::pow(d[39],2) - std::pow(d[41],2) + std::pow(d[42],2) - std::pow(d[44],2);
				coeffs[73] = 2*d[0]*d[48] - 2*d[2]*d[50] + 2*d[3]*d[51] - 2*d[5]*d[53] + 2*d[6]*d[54] - 2*d[8]*d[56];
				coeffs[74] = 2*d[12]*d[48] - 2*d[14]*d[50] + 2*d[15]*d[51] - 2*d[17]*d[53] + 2*d[18]*d[54] - 2*d[20]*d[56];
				coeffs[75] = 2*d[24]*d[48] - 2*d[26]*d[50] + 2*d[27]*d[51] - 2*d[29]*d[53] + 2*d[30]*d[54] - 2*d[32]*d[56];
				coeffs[76] = 2*d[36]*d[48] - 2*d[38]*d[50] + 2*d[39]*d[51] - 2*d[41]*d[53] + 2*d[42]*d[54] - 2*d[44]*d[56];
				coeffs[77] = std::pow(d[48],2) - std::pow(d[50],2) + std::pow(d[51],2) - std::pow(d[53],2) + std::pow(d[54],2) - std::pow(d[56],2);
				coeffs[78] = 2*d[0]*d[60] - 2*d[2]*d[62] + 2*d[3]*d[63] - 2*d[5]*d[65] + 2*d[6]*d[66] - 2*d[8]*d[68];
				coeffs[79] = 2*d[12]*d[60] - 2*d[14]*d[62] + 2*d[15]*d[63] - 2*d[17]*d[65] + 2*d[18]*d[66] - 2*d[20]*d[68];
				coeffs[80] = 2*d[24]*d[60] - 2*d[26]*d[62] + 2*d[27]*d[63] - 2*d[29]*d[65] + 2*d[30]*d[66] - 2*d[32]*d[68];
				coeffs[81] = 2*d[36]*d[60] - 2*d[38]*d[62] + 2*d[39]*d[63] - 2*d[41]*d[65] + 2*d[42]*d[66] - 2*d[44]*d[68];
				coeffs[82] = 2*d[48]*d[60] - 2*d[50]*d[62] + 2*d[51]*d[63] - 2*d[53]*d[65] + 2*d[54]*d[66] - 2*d[56]*d[68];
				coeffs[83] = std::pow(d[60],2) - std::pow(d[62],2) + std::pow(d[63],2) - std::pow(d[65],2) + std::pow(d[66],2) - std::pow(d[68],2);
				coeffs[84] = d[0]*d[3] + d[1]*d[4] + d[2]*d[5];
				coeffs[85] = d[3]*d[12] + d[4]*d[13] + d[5]*d[14] + d[0]*d[15] + d[1]*d[16] + d[2]*d[17];
				coeffs[86] = d[12]*d[15] + d[13]*d[16] + d[14]*d[17];
				coeffs[87] = d[3]*d[24] + d[4]*d[25] + d[5]*d[26] + d[0]*d[27] + d[1]*d[28] + d[2]*d[29];
				coeffs[88] = d[15]*d[24] + d[16]*d[25] + d[17]*d[26] + d[12]*d[27] + d[13]*d[28] + d[14]*d[29];
				coeffs[89] = d[24]*d[27] + d[25]*d[28] + d[26]*d[29];
				coeffs[90] = d[3]*d[36] + d[4]*d[37] + d[5]*d[38] + d[0]*d[39] + d[1]*d[40] + d[2]*d[41];
				coeffs[91] = d[15]*d[36] + d[16]*d[37] + d[17]*d[38] + d[12]*d[39] + d[13]*d[40] + d[14]*d[41];
				coeffs[92] = d[27]*d[36] + d[28]*d[37] + d[29]*d[38] + d[24]*d[39] + d[25]*d[40] + d[26]*d[41];
				coeffs[93] = d[36]*d[39] + d[37]*d[40] + d[38]*d[41];
				coeffs[94] = d[3]*d[48] + d[4]*d[49] + d[5]*d[50] + d[0]*d[51] + d[1]*d[52] + d[2]*d[53];
				coeffs[95] = d[15]*d[48] + d[16]*d[49] + d[17]*d[50] + d[12]*d[51] + d[13]*d[52] + d[14]*d[53];
				coeffs[96] = d[27]*d[48] + d[28]*d[49] + d[29]*d[50] + d[24]*d[51] + d[25]*d[52] + d[26]*d[53];
				coeffs[97] = d[39]*d[48] + d[40]*d[49] + d[41]*d[50] + d[36]*d[51] + d[37]*d[52] + d[38]*d[53];
				coeffs[98] = d[48]*d[51] + d[49]*d[52] + d[50]*d[53];
				coeffs[99] = d[3]*d[60] + d[4]*d[61] + d[5]*d[62] + d[0]*d[63] + d[1]*d[64] + d[2]*d[65];
				coeffs[100] = d[15]*d[60] + d[16]*d[61] + d[17]*d[62] + d[12]*d[63] + d[13]*d[64] + d[14]*d[65];
				coeffs[101] = d[27]*d[60] + d[28]*d[61] + d[29]*d[62] + d[24]*d[63] + d[25]*d[64] + d[26]*d[65];
				coeffs[102] = d[39]*d[60] + d[40]*d[61] + d[41]*d[62] + d[36]*d[63] + d[37]*d[64] + d[38]*d[65];
				coeffs[103] = d[51]*d[60] + d[52]*d[61] + d[53]*d[62] + d[48]*d[63] + d[49]*d[64] + d[50]*d[65];
				coeffs[104] = d[60]*d[63] + d[61]*d[64] + d[62]*d[65];
				coeffs[105] = d[0]*d[6] + d[1]*d[7] + d[2]*d[8];
				coeffs[106] = d[6]*d[12] + d[7]*d[13] + d[8]*d[14] + d[0]*d[18] + d[1]*d[19] + d[2]*d[20];
				coeffs[107] = d[12]*d[18] + d[13]*d[19] + d[14]*d[20];
				coeffs[108] = d[6]*d[24] + d[7]*d[25] + d[8]*d[26] + d[0]*d[30] + d[1]*d[31] + d[2]*d[32];
				coeffs[109] = d[18]*d[24] + d[19]*d[25] + d[20]*d[26] + d[12]*d[30] + d[13]*d[31] + d[14]*d[32];
				coeffs[110] = d[24]*d[30] + d[25]*d[31] + d[26]*d[32];
				coeffs[111] = d[6]*d[36] + d[7]*d[37] + d[8]*d[38] + d[0]*d[42] + d[1]*d[43] + d[2]*d[44];
				coeffs[112] = d[18]*d[36] + d[19]*d[37] + d[20]*d[38] + d[12]*d[42] + d[13]*d[43] + d[14]*d[44];
				coeffs[113] = d[30]*d[36] + d[31]*d[37] + d[32]*d[38] + d[24]*d[42] + d[25]*d[43] + d[26]*d[44];
				coeffs[114] = d[36]*d[42] + d[37]*d[43] + d[38]*d[44];
				coeffs[115] = d[6]*d[48] + d[7]*d[49] + d[8]*d[50] + d[0]*d[54] + d[1]*d[55] + d[2]*d[56];
				coeffs[116] = d[18]*d[48] + d[19]*d[49] + d[20]*d[50] + d[12]*d[54] + d[13]*d[55] + d[14]*d[56];
				coeffs[117] = d[30]*d[48] + d[31]*d[49] + d[32]*d[50] + d[24]*d[54] + d[25]*d[55] + d[26]*d[56];
				coeffs[118] = d[42]*d[48] + d[43]*d[49] + d[44]*d[50] + d[36]*d[54] + d[37]*d[55] + d[38]*d[56];
				coeffs[119] = d[48]*d[54] + d[49]*d[55] + d[50]*d[56];
				coeffs[120] = d[6]*d[60] + d[7]*d[61] + d[8]*d[62] + d[0]*d[66] + d[1]*d[67] + d[2]*d[68];
				coeffs[121] = d[18]*d[60] + d[19]*d[61] + d[20]*d[62] + d[12]*d[66] + d[13]*d[67] + d[14]*d[68];
				coeffs[122] = d[30]*d[60] + d[31]*d[61] + d[32]*d[62] + d[24]*d[66] + d[25]*d[67] + d[26]*d[68];
				coeffs[123] = d[42]*d[60] + d[43]*d[61] + d[44]*d[62] + d[36]*d[66] + d[37]*d[67] + d[38]*d[68];
				coeffs[124] = d[54]*d[60] + d[55]*d[61] + d[56]*d[62] + d[48]*d[66] + d[49]*d[67] + d[50]*d[68];
				coeffs[125] = d[60]*d[66] + d[61]*d[67] + d[62]*d[68];
				coeffs[126] = d[3]*d[6] + d[4]*d[7] + d[5]*d[8];
				coeffs[127] = d[6]*d[15] + d[7]*d[16] + d[8]*d[17] + d[3]*d[18] + d[4]*d[19] + d[5]*d[20];
				coeffs[128] = d[15]*d[18] + d[16]*d[19] + d[17]*d[20];
				coeffs[129] = d[6]*d[27] + d[7]*d[28] + d[8]*d[29] + d[3]*d[30] + d[4]*d[31] + d[5]*d[32];
				coeffs[130] = d[18]*d[27] + d[19]*d[28] + d[20]*d[29] + d[15]*d[30] + d[16]*d[31] + d[17]*d[32];
				coeffs[131] = d[27]*d[30] + d[28]*d[31] + d[29]*d[32];
				coeffs[132] = d[6]*d[39] + d[7]*d[40] + d[8]*d[41] + d[3]*d[42] + d[4]*d[43] + d[5]*d[44];
				coeffs[133] = d[18]*d[39] + d[19]*d[40] + d[20]*d[41] + d[15]*d[42] + d[16]*d[43] + d[17]*d[44];
				coeffs[134] = d[30]*d[39] + d[31]*d[40] + d[32]*d[41] + d[27]*d[42] + d[28]*d[43] + d[29]*d[44];
				coeffs[135] = d[39]*d[42] + d[40]*d[43] + d[41]*d[44];
				coeffs[136] = d[6]*d[51] + d[7]*d[52] + d[8]*d[53] + d[3]*d[54] + d[4]*d[55] + d[5]*d[56];
				coeffs[137] = d[18]*d[51] + d[19]*d[52] + d[20]*d[53] + d[15]*d[54] + d[16]*d[55] + d[17]*d[56];
				coeffs[138] = d[30]*d[51] + d[31]*d[52] + d[32]*d[53] + d[27]*d[54] + d[28]*d[55] + d[29]*d[56];
				coeffs[139] = d[42]*d[51] + d[43]*d[52] + d[44]*d[53] + d[39]*d[54] + d[40]*d[55] + d[41]*d[56];
				coeffs[140] = d[51]*d[54] + d[52]*d[55] + d[53]*d[56];
				coeffs[141] = d[6]*d[63] + d[7]*d[64] + d[8]*d[65] + d[3]*d[66] + d[4]*d[67] + d[5]*d[68];
				coeffs[142] = d[18]*d[63] + d[19]*d[64] + d[20]*d[65] + d[15]*d[66] + d[16]*d[67] + d[17]*d[68];
				coeffs[143] = d[30]*d[63] + d[31]*d[64] + d[32]*d[65] + d[27]*d[66] + d[28]*d[67] + d[29]*d[68];
				coeffs[144] = d[42]*d[63] + d[43]*d[64] + d[44]*d[65] + d[39]*d[66] + d[40]*d[67] + d[41]*d[68];
				coeffs[145] = d[54]*d[63] + d[55]*d[64] + d[56]*d[65] + d[51]*d[66] + d[52]*d[67] + d[53]*d[68];
				coeffs[146] = d[63]*d[66] + d[64]*d[67] + d[65]*d[68];
				coeffs[147] = d[0]*d[1] + d[3]*d[4] + d[6]*d[7];
				coeffs[148] = d[1]*d[12] + d[0]*d[13] + d[4]*d[15] + d[3]*d[16] + d[7]*d[18] + d[6]*d[19];
				coeffs[149] = d[12]*d[13] + d[15]*d[16] + d[18]*d[19];
				coeffs[150] = d[1]*d[24] + d[0]*d[25] + d[4]*d[27] + d[3]*d[28] + d[7]*d[30] + d[6]*d[31];
				coeffs[151] = d[13]*d[24] + d[12]*d[25] + d[16]*d[27] + d[15]*d[28] + d[19]*d[30] + d[18]*d[31];
				coeffs[152] = d[24]*d[25] + d[27]*d[28] + d[30]*d[31];
				coeffs[153] = d[1]*d[36] + d[0]*d[37] + d[4]*d[39] + d[3]*d[40] + d[7]*d[42] + d[6]*d[43];
				coeffs[154] = d[13]*d[36] + d[12]*d[37] + d[16]*d[39] + d[15]*d[40] + d[19]*d[42] + d[18]*d[43];
				coeffs[155] = d[25]*d[36] + d[24]*d[37] + d[28]*d[39] + d[27]*d[40] + d[31]*d[42] + d[30]*d[43];
				coeffs[156] = d[36]*d[37] + d[39]*d[40] + d[42]*d[43];
				coeffs[157] = d[1]*d[48] + d[0]*d[49] + d[4]*d[51] + d[3]*d[52] + d[7]*d[54] + d[6]*d[55];
				coeffs[158] = d[13]*d[48] + d[12]*d[49] + d[16]*d[51] + d[15]*d[52] + d[19]*d[54] + d[18]*d[55];
				coeffs[159] = d[25]*d[48] + d[24]*d[49] + d[28]*d[51] + d[27]*d[52] + d[31]*d[54] + d[30]*d[55];
				coeffs[160] = d[37]*d[48] + d[36]*d[49] + d[40]*d[51] + d[39]*d[52] + d[43]*d[54] + d[42]*d[55];
				coeffs[161] = d[48]*d[49] + d[51]*d[52] + d[54]*d[55];
				coeffs[162] = d[1]*d[60] + d[0]*d[61] + d[4]*d[63] + d[3]*d[64] + d[7]*d[66] + d[6]*d[67];
				coeffs[163] = d[13]*d[60] + d[12]*d[61] + d[16]*d[63] + d[15]*d[64] + d[19]*d[66] + d[18]*d[67];
				coeffs[164] = d[25]*d[60] + d[24]*d[61] + d[28]*d[63] + d[27]*d[64] + d[31]*d[66] + d[30]*d[67];
				coeffs[165] = d[37]*d[60] + d[36]*d[61] + d[40]*d[63] + d[39]*d[64] + d[43]*d[66] + d[42]*d[67];
				coeffs[166] = d[49]*d[60] + d[48]*d[61] + d[52]*d[63] + d[51]*d[64] + d[55]*d[66] + d[54]*d[67];
				coeffs[167] = d[60]*d[61] + d[63]*d[64] + d[66]*d[67];
				coeffs[168] = d[0]*d[2] + d[3]*d[5] + d[6]*d[8];
				coeffs[169] = d[2]*d[12] + d[0]*d[14] + d[5]*d[15] + d[3]*d[17] + d[8]*d[18] + d[6]*d[20];
				coeffs[170] = d[12]*d[14] + d[15]*d[17] + d[18]*d[20];
				coeffs[171] = d[2]*d[24] + d[0]*d[26] + d[5]*d[27] + d[3]*d[29] + d[8]*d[30] + d[6]*d[32];
				coeffs[172] = d[14]*d[24] + d[12]*d[26] + d[17]*d[27] + d[15]*d[29] + d[20]*d[30] + d[18]*d[32];
				coeffs[173] = d[24]*d[26] + d[27]*d[29] + d[30]*d[32];
				coeffs[174] = d[2]*d[36] + d[0]*d[38] + d[5]*d[39] + d[3]*d[41] + d[8]*d[42] + d[6]*d[44];
				coeffs[175] = d[14]*d[36] + d[12]*d[38] + d[17]*d[39] + d[15]*d[41] + d[20]*d[42] + d[18]*d[44];
				coeffs[176] = d[26]*d[36] + d[24]*d[38] + d[29]*d[39] + d[27]*d[41] + d[32]*d[42] + d[30]*d[44];
				coeffs[177] = d[36]*d[38] + d[39]*d[41] + d[42]*d[44];
				coeffs[178] = d[2]*d[48] + d[0]*d[50] + d[5]*d[51] + d[3]*d[53] + d[8]*d[54] + d[6]*d[56];
				coeffs[179] = d[14]*d[48] + d[12]*d[50] + d[17]*d[51] + d[15]*d[53] + d[20]*d[54] + d[18]*d[56];
				coeffs[180] = d[26]*d[48] + d[24]*d[50] + d[29]*d[51] + d[27]*d[53] + d[32]*d[54] + d[30]*d[56];
				coeffs[181] = d[38]*d[48] + d[36]*d[50] + d[41]*d[51] + d[39]*d[53] + d[44]*d[54] + d[42]*d[56];
				coeffs[182] = d[48]*d[50] + d[51]*d[53] + d[54]*d[56];
				coeffs[183] = d[2]*d[60] + d[0]*d[62] + d[5]*d[63] + d[3]*d[65] + d[8]*d[66] + d[6]*d[68];
				coeffs[184] = d[14]*d[60] + d[12]*d[62] + d[17]*d[63] + d[15]*d[65] + d[20]*d[66] + d[18]*d[68];
				coeffs[185] = d[26]*d[60] + d[24]*d[62] + d[29]*d[63] + d[27]*d[65] + d[32]*d[66] + d[30]*d[68];
				coeffs[186] = d[38]*d[60] + d[36]*d[62] + d[41]*d[63] + d[39]*d[65] + d[44]*d[66] + d[42]*d[68];
				coeffs[187] = d[50]*d[60] + d[48]*d[62] + d[53]*d[63] + d[51]*d[65] + d[56]*d[66] + d[54]*d[68];
				coeffs[188] = d[60]*d[62] + d[63]*d[65] + d[66]*d[68];
				coeffs[189] = d[1]*d[2] + d[4]*d[5] + d[7]*d[8];
				coeffs[190] = d[2]*d[13] + d[1]*d[14] + d[5]*d[16] + d[4]*d[17] + d[8]*d[19] + d[7]*d[20];
				coeffs[191] = d[13]*d[14] + d[16]*d[17] + d[19]*d[20];
				coeffs[192] = d[2]*d[25] + d[1]*d[26] + d[5]*d[28] + d[4]*d[29] + d[8]*d[31] + d[7]*d[32];
				coeffs[193] = d[14]*d[25] + d[13]*d[26] + d[17]*d[28] + d[16]*d[29] + d[20]*d[31] + d[19]*d[32];
				coeffs[194] = d[25]*d[26] + d[28]*d[29] + d[31]*d[32];
				coeffs[195] = d[2]*d[37] + d[1]*d[38] + d[5]*d[40] + d[4]*d[41] + d[8]*d[43] + d[7]*d[44];
				coeffs[196] = d[14]*d[37] + d[13]*d[38] + d[17]*d[40] + d[16]*d[41] + d[20]*d[43] + d[19]*d[44];
				coeffs[197] = d[26]*d[37] + d[25]*d[38] + d[29]*d[40] + d[28]*d[41] + d[32]*d[43] + d[31]*d[44];
				coeffs[198] = d[37]*d[38] + d[40]*d[41] + d[43]*d[44];
				coeffs[199] = d[2]*d[49] + d[1]*d[50] + d[5]*d[52] + d[4]*d[53] + d[8]*d[55] + d[7]*d[56];
				coeffs[200] = d[14]*d[49] + d[13]*d[50] + d[17]*d[52] + d[16]*d[53] + d[20]*d[55] + d[19]*d[56];
				coeffs[201] = d[26]*d[49] + d[25]*d[50] + d[29]*d[52] + d[28]*d[53] + d[32]*d[55] + d[31]*d[56];
				coeffs[202] = d[38]*d[49] + d[37]*d[50] + d[41]*d[52] + d[40]*d[53] + d[44]*d[55] + d[43]*d[56];
				coeffs[203] = d[49]*d[50] + d[52]*d[53] + d[55]*d[56];
				coeffs[204] = d[2]*d[61] + d[1]*d[62] + d[5]*d[64] + d[4]*d[65] + d[8]*d[67] + d[7]*d[68];
				coeffs[205] = d[14]*d[61] + d[13]*d[62] + d[17]*d[64] + d[16]*d[65] + d[20]*d[67] + d[19]*d[68];
				coeffs[206] = d[26]*d[61] + d[25]*d[62] + d[29]*d[64] + d[28]*d[65] + d[32]*d[67] + d[31]*d[68];
				coeffs[207] = d[38]*d[61] + d[37]*d[62] + d[41]*d[64] + d[40]*d[65] + d[44]*d[67] + d[43]*d[68];
				coeffs[208] = d[50]*d[61] + d[49]*d[62] + d[53]*d[64] + d[52]*d[65] + d[56]*d[67] + d[55]*d[68];
				coeffs[209] = d[61]*d[62] + d[64]*d[65] + d[67]*d[68];



				// Setup elimination template
				static const int coeffs0_ind[] = { 0,21,42,63,84,105,126,1,22,43,64,85,106,127,2,23,44,65,86,107,128,0,21,42,63,84,105,126,147,168,189,3,24,45,1,22,43,64,66,85,87,106,108,127,129,148,169,190,4,25,46,2,23,44,65,67,86,88,107,109,128,130,149,170,191,3,24,45,66,87,108,129,150,171,192,5,26,47,4,25,46,67,68,88,89,109,110,130,131,151,172,193,5,26,47,68,89,110,131,152,173,194,42,0,63,21,84,105,126,147,168,189,6,27,48,69,43,1,64,22,85,90,106,111,127,132,148,169,190,7,28,49,70,44,2,65,23,86,91,107,112,128,133,149,170,191,6,27,48,69,90,45,3,66,24,87,108,111,129,132,150,153,171,174,192,195,8,29,50,7,28,49,70,71,91,46,4,67,25,88,92,109,112,113,130,133,134,151,154,172,175,193,196,8,29,50,71,92,47,5,68,26,89,110,113,131,134,152,155,173,176,194,197,48,6,69,27,90,111,132,153,174,195,9,30,51,72,49,7,70,28,91,93,112,114,133,135,154,175,196,9,30,51,72,93,50,8,71,29,92,113,114,134,135,155,156,176,177,197,198,51,9,72,30,93,114,135,156,177,198,0,84,105,63,42,126,21,147,168,189,10,31,52,73,94,115,1,85,106,64,43,127,22,136,148,169,190,11,32,53,74,95,116,2,86,107,65,44,128,23,137,149,170,191,10,31,52,73,94,115,3,87,108,66,45,129,24,136,150,157,171,178,192,199,12,33,54,11,32,53,74,75,95,96,116,117,4,88,109,67,46,130,25,137,138,151,158,172,179,193,200,12,33,54,75,96,117,5,89,110,68,47,131,26,138,152,159,173,180,194,201,52,10,73,31,94,115,136,6,90,111,69,48,132,27,153,157,174,178,195,199,13,34,55,76,53,11,74,32,95,97,116,118,137,7,91,112,70,49,133,28,139,154,158,175,179,196,200,13,34,55,76,97,54,12,75,33,96,117,118,138,8,92,113,71,50,134,29,139,155,159,160,176,180,181,197,201,202,55,13,76,34,97,118,139,9,93,114,72,51,135,30,156,160,177,181,198,202,10,94,115,73,52,136,31,157,178,199,14,35,56,77,98,119,11,95,116,74,53,137,32,140,158,179,200,14,35,56,77,98,119,12,96,117,75,54,138,33,140,159,161,180,182,201,203,105,84,63,147,21,0,126,168,42,189,15,36,57,78,99,120,141,106,85,64,148,22,1,127,169,43,190,16,37,58,79,100,121,142,107,86,65,149,23,2,128,170,44,191,15,36,57,78,99,120,141,162,108,87,66,150,24,3,129,171,45,183,192,204,17,38,59,16,37,58,79,80,100,101,121,122,142,143,163,109,88,67,151,25,4,130,172,46,184,193,205,17,38,59,80,101,122,143,164,110,89,68,152,26,5,131,173,47,185,194,206,57,15,78,36,99,120,141,162,111,90,69,153,27,6,132,174,48,183,195,204,18,39,60,81,58,16,79,37,100,102,121,123,142,144,163,112,91,70,154,28,7,133,175,49,184,196,205,18,39,60,81,102,59,17,80,38,101,122,123,143,144,164,165,113,92,71,155,29,8,134,176,50,185,186,197,206,207,60,18,81,39,102,123,144,165,114,93,72,156,30,9,135,177,51,186,198,207,15,99,120,78,57,141,36,162,183,115,94,73,157,31,10,136,178,52,199,204,19,40,61,82,103,124,16,100,121,79,58,142,37,145,163,184,116,95,74,158,32,11,137,179,53,200,205,19,40,61,82,103,124,17,101,122,80,59,143,38,145,164,166,185,117,96,75,159,33,12,138,180,54,187,201,206,208,56,14,77,35,98,119,140,13,97,118,76,55,139,34,160,161,181,182,202,203,14,98,119,77,56,140,35,161,182,203 };
				static const int coeffs1_ind[] = { 125,104,83,167,41,20,146,188,62,209,120,99,78,162,36,15,141,183,57,204,20,41,62,83,104,125,146,121,100,79,163,37,16,142,184,58,205,20,41,62,83,104,125,146,167,122,101,80,164,38,17,143,185,59,188,206,209,62,20,83,41,104,125,146,167,123,102,81,165,39,18,144,186,60,188,207,209,61,19,82,40,103,124,145,18,102,123,81,60,144,39,165,166,186,118,97,76,160,34,13,139,181,55,187,202,207,208,20,104,125,83,62,146,41,167,188,124,103,82,166,40,19,145,187,61,208,209,19,103,124,82,61,145,40,166,187,119,98,77,161,35,14,140,182,56,203,208 };
				static const int C0_ind[] = { 0,1,2,7,14,17,27,47,48,49,54,61,64,74,94,95,96,101,108,111,121,144,145,146,147,149,157,167,171,183,187,188,189,190,191,192,193,194,195,196,202,204,205,214,215,218,230,234,235,236,237,238,239,240,241,242,243,249,251,252,261,262,265,277,281,285,286,287,288,290,298,308,312,324,328,329,330,331,332,333,334,335,336,337,343,345,346,355,356,359,371,375,379,380,381,382,384,392,402,406,418,422,432,433,434,435,436,438,441,452,464,468,470,471,472,477,479,480,481,482,483,484,485,487,488,497,499,511,515,517,518,519,524,526,527,528,529,530,531,532,534,535,544,546,558,562,567,568,569,570,572,573,574,575,576,577,579,580,582,590,593,594,605,606,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,637,638,640,641,652,653,656,657,661,662,663,664,666,667,668,669,670,671,673,674,676,684,687,688,699,700,703,704,714,715,716,717,718,720,723,734,746,750,752,753,754,759,761,762,763,764,765,766,767,769,770,779,781,793,797,802,803,804,805,807,808,809,810,811,812,814,815,817,825,828,829,840,841,844,845,855,856,857,858,859,861,864,875,887,891,912,913,914,915,916,917,918,921,924,937,940,941,942,947,954,957,959,960,961,962,963,964,965,967,968,971,984,987,988,989,994,1001,1004,1006,1007,1008,1009,1010,1011,1012,1014,1015,1018,1031,1037,1038,1039,1040,1042,1050,1053,1054,1055,1056,1057,1058,1059,1060,1062,1064,1065,1076,1078,1080,1081,1082,1083,1084,1085,1086,1087,1088,1089,1095,1097,1098,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1111,1112,1123,1125,1127,1131,1132,1133,1134,1136,1144,1147,1148,1149,1150,1151,1152,1153,1154,1156,1158,1159,1170,1172,1174,1184,1185,1186,1187,1188,1190,1193,1194,1195,1196,1197,1198,1199,1200,1203,1204,1206,1216,1219,1220,1222,1223,1224,1229,1231,1232,1233,1234,1235,1236,1237,1239,1240,1241,1242,1243,1244,1245,1246,1247,1249,1250,1251,1253,1263,1266,1267,1272,1273,1274,1275,1277,1278,1279,1280,1281,1282,1284,1285,1287,1288,1289,1290,1291,1292,1293,1294,1295,1297,1298,1299,1300,1310,1311,1313,1314,1315,1325,1326,1327,1328,1329,1331,1334,1335,1336,1337,1338,1339,1340,1341,1344,1345,1347,1357,1360,1361,1382,1383,1384,1385,1386,1387,1388,1391,1394,1407,1410,1411,1412,1417,1424,1427,1429,1430,1431,1432,1433,1434,1435,1437,1438,1441,1454,1460,1461,1462,1463,1465,1473,1476,1477,1478,1479,1480,1481,1482,1483,1485,1487,1488,1499,1501,1503,1536,1537,1538,1539,1540,1541,1542,1543,1544,1547,1551,1552,1553,1558,1565,1568,1578,1583,1584,1585,1586,1587,1588,1589,1590,1591,1594,1598,1599,1600,1605,1612,1615,1625,1630,1631,1632,1633,1634,1635,1636,1637,1638,1641,1648,1649,1650,1651,1653,1661,1671,1675,1677,1678,1679,1680,1681,1682,1683,1684,1685,1687,1688,1691,1692,1693,1694,1695,1696,1697,1698,1699,1700,1706,1708,1709,1718,1719,1722,1724,1725,1726,1727,1728,1729,1730,1731,1732,1734,1735,1738,1742,1743,1744,1745,1747,1755,1765,1769,1771,1772,1773,1774,1775,1776,1777,1778,1779,1781,1782,1785,1795,1796,1797,1798,1799,1801,1804,1815,1818,1819,1820,1821,1822,1823,1824,1825,1826,1827,1829,1831,1833,1834,1835,1840,1842,1843,1844,1845,1846,1847,1848,1850,1851,1860,1862,1865,1866,1867,1868,1869,1870,1871,1872,1873,1874,1876,1878,1883,1884,1885,1886,1888,1889,1890,1891,1892,1893,1895,1896,1898,1906,1909,1910,1912,1913,1914,1915,1916,1917,1918,1919,1920,1921,1922,1923,1925,1926,1936,1937,1938,1939,1940,1942,1945,1956,1959,1960,1961,1962,1963,1964,1965,1966,1967,1968,1970,1972,1993,1994,1995,1996,1997,1998,1999,2002,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2017,2018,2021,2022,2023,2028,2035,2038,2040,2041,2042,2043,2044,2045,2046,2048,2049,2052,2053,2054,2055,2056,2057,2058,2059,2060,2061,2064,2065,2071,2072,2073,2074,2076,2084,2087,2088,2089,2090,2091,2092,2093,2094,2096,2098,2099,2100,2101,2102,2103,2104,2105,2106,2107,2108,2110,2111,2112,2114,2124,2125,2126,2127,2128,2130,2133,2134,2135,2136,2137,2138,2139,2140,2143,2144,2146,2156,2159,2160,2181,2182,2183,2184,2185,2186,2187,2190,2193,2206 } ;
				static const int C1_ind[] = { 32,33,34,35,36,37,38,39,40,43,79,80,81,82,83,84,85,86,87,90,94,95,96,101,108,111,121,126,127,128,129,130,131,132,133,134,137,144,145,146,147,149,157,167,171,173,174,175,176,177,178,179,180,181,183,184,187,197,198,199,200,201,203,206,217,220,221,222,223,224,225,226,227,228,229,231,233,244,245,246,247,248,250,253,254,255,256,257,258,259,260,263,264,266,267,268,269,270,271,272,273,274,275,276,278,279,280,301,302,303,304,305,306,307,310,313,314,315,316,317,318,319,320,321,322,325,326,348,349,350,351,352,353,354,357,360,361,362,363,364,365,366,367,368,369,372,373 };

				Eigen::Matrix<double,47,47> C0; C0.setZero();
				Eigen::Matrix<double,47,8> C1; C1.setZero();
				for (int i = 0; i < 840; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
				for (int i = 0; i < 147; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

				Eigen::Matrix<double,47,8> C12 = C0.partialPivLu().solve(C1);




				// Setup action matrix
				Eigen::Matrix<double,13, 8> RR;
				RR << -C12.bottomRows(5), Eigen::Matrix<double,8,8>::Identity(8, 8);

				static const int AM_ind[] = { 11,0,1,2,10,3,12,4 };
				Eigen::Matrix<double, 8, 8> AM;
				for (int i = 0; i < 8; i++) {
					AM.row(i) = RR.row(AM_ind[i]);
				}

				Eigen::Matrix<std::complex<double>, 5, 8> sols;
				sols.setZero();

				// Solve eigenvalue problem
				Eigen::EigenSolver<Eigen::Matrix<double, 8, 8> > es(AM);
				Eigen::ArrayXcd D = es.eigenvalues();
				Eigen::ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).array().replicate(8, 1)).eval();

				sols.row(0) = V.row(1).array();
				sols.row(1) = V.row(2).array();
				sols.row(2) = V.row(3).array();
				sols.row(3) = V.row(4).array();
				sols.row(4) = D.transpose().array();
				return sols;
			}
		}
	}
}