// rpoly_ak1.cpp - Program for calculating the roots of a polynomial of real coefficients.
// Written in Visual C++ 2005 Express Edition
// 14 July 2007
//
// The sub-routines listed below are translations of the FORTRAN routines included in RPOLY.FOR,
// posted off the NETLIB site as TOMS/493:
//
// http://www.netlib.org/toms/493
//
// TOMS/493 is based on the Jenkins-Traub algorithm.
//
// To distinguish the routines posted below from others, an _ak1 suffix has been appended to them.
//
// Following is a list of the major changes made in the course of translating the TOMS/493 routines
// to the C++ versions posted below:
// 1) All global variables have been eliminated.
// 2) The "FAIL" parameter passed into RPOLY.FOR has been eliminated.
// 3) RPOLY.FOR solves polynomials of degree up to 100, but does not explicitly state this limit.
//     rpoly_ak1 explicitly states this limit; uses the macro name MAXDEGREE to specify this limit;
//     and does a check to ensure that the user input variable Degree is not greater than MAXDEGREE
//     (if it is, an error message is output and rpoly_ak1 terminates). If a user wishes to compute
//     roots of polynomials of degree greater than MAXDEGREE, using a macro name like MAXDEGREE provides
//     the simplest way of offering this capability.
// 4) All "GO TO" statements have been eliminated.
//
// A small main program is included also, to provide an example of how to use rpoly_ak1. In this
// example, data is input from a file to eliminate the need for a user to type data in via
// the console.

#ifndef __RPOLY_H__
#define __RPOLY_H__

#define MAXDEGREE 100
#define MDP1 MAXDEGREE+1

void rpoly_ak1(double op[MDP1], int* Degree, double zeror[MAXDEGREE], double zeroi[MAXDEGREE]);
void Fxshfr_ak1(int L2, int* NZ, double sr, double v, double K[MDP1], int N, double p[MDP1], int NN, double qp[MDP1], double u, double* lzi, double* lzr, double* szi, double* szr);
void QuadSD_ak1(int NN, double u, double v, double p[MDP1], double q[MDP1], double* a, double* b);
int calcSC_ak1(int N, double a, double b, double* a1, double* a3, double* a7, double* c, double* d, double* e, double* f, double* g, double* h, double K[MDP1], double u, double v, double qk[MDP1]);
void nextK_ak1(int N, int tFlag, double a, double b, double a1, double* a3, double* a7, double K[MDP1], double qk[MDP1], double qp[MDP1]);
void newest_ak1(int tFlag, double* uu, double* vv, double a, double a1, double a3, double a7, double b, double c, double d, double f, double g, double h, double u, double v, double K[MDP1], int N, double p[MDP1]);
void QuadIT_ak1(int N, int* NZ, double uu, double vv, double* szr, double* szi, double* lzr, double* lzi, double qp[MDP1], int NN, double* a, double* b, double p[MDP1], double qk[MDP1], double* a1, double* a3, double* a7, double* c, double* d, double* e, double* f, double* g, double* h, double K[MDP1]);
void RealIT_ak1(int* iFlag, int* NZ, double* sss, int N, double p[MDP1], int NN, double qp[MDP1], double* szr, double* szi, double K[MDP1], double qk[MDP1]);
void Quad_ak1(double a, double b1, double c, double* sr, double* si, double* lr, double* li);

#endif
