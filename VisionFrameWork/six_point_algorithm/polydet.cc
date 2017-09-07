// Copyright Richard Hartley, 2010
static const char *copyright = "Copyright Richard Hartley, 2010";

//--------------------------------------------------------------------------
// LICENSE INFORMATION
//
// 1.  For academic/research users:
//
// This program is free for academic/research purpose:   you can redistribute
// it and/or modify  it under the terms of the GNU General Public License as 
// published by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Under this academic/research condition,  this program is distributed in 
// the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program. If not, see <http://www.gnu.org/licenses/>.
//
// 2.  For commercial OEMs, ISVs and VARs:
// 
// For OEMs, ISVs, and VARs who distribute/modify/use this software 
// (binaries or source code) with their products, and do not license and 
// distribute their source code under the GPL, please contact NICTA 
// (www.nicta.com.au), and NICTA will provide a flexible OEM Commercial 
// License. 
//
//---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hidden6.h"

// #define RH_DEBUG

// For generating synthetic polynomials
const int ColumnDegree[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

const int NRepetitions = 10000;

int stuffit () { return 1; }

static double urandom()
   {
   // Returns a real random between -1 and 1
   const int MAXRAND = 65000;
   return 4.0*((rand()%MAXRAND)/((double) MAXRAND) - 0.5);
   // return rand() % 20 - 10.0;
   }

void eval_poly (PolyMatrix &Q, PolyDegree &deg, double x)
   {
   // Evaluates the polynomial at a given value, overwriting it
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         // Evaluate the poly
         for (int k=deg[i][j]-1; k>=0; k--)
            Q[i][j][k] += x*Q[i][j][k+1];

         // Set degree to zero
         deg[i][j] = 0;
         }
   }

void cross_prod (
                 double a11[], int deg11, 
                 double a12[], int deg12,
                 double a21[], int deg21,
                 double a22[], int deg22,
                 double a00[], double toep[], int deg00,
                 double res[], int &dres,
		 BMatrix B, int &current_size
		 )
   {
   // Does a single 2x2 cross multiplication

   // Do the multiplcation in temporary storage
   double temp[2*Maxdegree + 1];

   // Work out the actual degree
   int deg1 = deg11 + deg22;
   int deg2 = deg12 + deg21;
   int deg = (deg1 > deg2) ? deg1 : deg2;

   // Clear out the temporary
   memset (temp, 0, sizeof(temp));

   // Now, start multiplying
   for (int i=0; i<=deg11; i++)
      for (int j=0; j<=deg22; j++)
         temp[i+j] += a11[i]*a22[j];

   for (int i=0; i<=deg12; i++)
      for (int j=0; j<=deg21; j++)
         temp[i+j] -= a12[i]*a21[j];

   // Clear out the result -- not really necessary
   memset (res, 0, (Maxdegree+1)*sizeof(double));

   //-----------------------------------------------------
   // This is the most tricky part of the code, to divide
   // one polynomial into the other.  By theory, the division
   // should be exact, but it is not, because of roundoff error.
   // we need to find a way to do this efficiently and accurately.
   //-----------------------------------------------------

#define USE_TOEPLITZ
#ifdef USE_TOEPLITZ

   // Now, divide by a00 - there should be no remainder
   int sres;
   polyquotient (temp, deg+1, a00, toep, deg00+1, res, sres, B, current_size);
   dres = sres-1;

#else

   // Now, divide by a00 - there should be no remainder
   double *pres = &(res[deg-deg00]);
   for (int d=deg; d>=deg00; d--)
      {
      // Work out the divisor
      int td = d - deg00;	// Degree of current term
      double val = temp[d] / a00[deg00];
      *(pres--) = val;

      // Do the subtraction involved in the division
      for (int j=0; j<deg00; j++)
         temp[j+td] -= val * a00[j];
      }
#endif

#ifdef RH_DEBUG
   // Print the remainder
   printf ("Remainder\n");
   for (int i=0; i<deg00; i++)
      printf ("\t%.5e\n", temp[i]);

#endif

   // Set the degree of the term
   dres = deg - deg00;
   }

void det_preprocess_6pt (
	PolyMatrix &Q, 
	PolyDegree degree, 
        int n_zero_roots	// Number of roots known to be zero
	)
   {
   // We do row-echelon form decomposition on the matrix to eliminate the
   // trivial known roots.
   // What is assumed here is the following.
   //   - the first row of the matrix consists of constants
   //   - the nullity of the matrix of constant terms is n_zero_roots,
   //     so when it is put in row-echelon form, the last n_zero_roots are zero.

   // Initialize the list of and columns.  We will do complete pivoting
   const int nrows = Nrows - 1;
   const int ncols = Nrows;

   int rows[Nrows], cols[Nrows];
   for (int i=0; i<nrows; i++) rows[i] = i+1;	// Miss the first row
   for (int i=0; i<ncols; i++) cols[i] = i;

   // Eliminate one row at a time
   for (int nr=nrows-1, nc=ncols-1; nr>=n_zero_roots; nr--,nc--)
      {
      // We must take the first row first to pivot around
      double bestval = 0.0;
      int bestrow = 0, bestcol = 0;

      // Find the highest value to pivot around
      for (int i=0; i<=nr; i++)
         for (int j=0; j<=nc; j++)
            {
            double val=Q[rows[i]][cols[j]][0];
            if (fabs(val) > bestval) 
               {
               bestval = fabs(val);
               bestrow = i;   // Actually rows[i]
               bestcol = j;
               }
            }

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
      // Print out the best value
      printf ("Pivot %d = %e at position %d %d\n",nr, 
	bestval, rows[bestrow], cols[bestcol]);
#endif

      // Now, select this row as a pivot.  Also keep track of rows pivoted
      int prow = rows[bestrow];
      rows[bestrow] = rows[nr];   // Replace pivot row by last row
      rows[nr] = prow;

      int pcol = cols[bestcol];
      cols[bestcol] = cols[nc];
      cols[nc] = pcol;

      // Clear out all the values above and to the right
      for (int i=0; i<nr; i++)
         {
         int iii = rows[i];
         double fac = Q[iii][pcol][0] / Q[prow][pcol][0];
         
         // Must do this to all the columns
         for (int j=0; j<ncols; j++)
            {
            int jjj = cols[j];
            int deg = degree[prow][jjj];
            int dij = degree[iii][jjj];
            if (deg>dij) degree[iii][jjj] = deg;
            for (int d=0; d<=deg; d++)
               {
               if (d <= dij)
                  Q[iii][jjj][d] -= Q[prow][jjj][d] * fac;
               else
                  Q[iii][jjj][d] = -Q[prow][jjj][d] * fac;
	       }
            }
         }
      }

   // Decrease the degree of the remaining rows
   for (int i=0; i<n_zero_roots; i++)
      {
      int ii = rows[i];
      for (int jj=0; jj<ncols; jj++)
         {
         // Decrease the degree of this element by one
         for (int d=1; d<=degree[ii][jj]; d++)
            Q[ii][jj][d-1] = Q[ii][jj][d];

         degree[ii][jj] -= 1;
         }
      }

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
   printf ("Degrees\n");
   for (int i=0; i<Nrows; i++)
      {
      for (int j=0; j<Nrows; j++)
         printf ("%1d ", degree[i][j]);
      printf ("\n");
      }
   printf("\n");

   printf ("Equation matrix\n");
   for (int i=0; i<nrows; i++)
       {
       for (int j=0; j<ncols; j++)
           printf ("%7.4f ", Q[rows[i]][cols[j]][0]);
       printf ("\n");
       }
   printf ("\n");
#endif

   }

double quick_compute_determinant (double A[Nrows][Nrows], int dim)
   {
   // Do row reduction on A to find the determinant (up to sign)

   // Initialize the list of rows
   int rows[Nrows];
   for (int i=0; i<dim; i++) rows[i] = i;

   // To accumulate the determinant
   double sign = 1.0;

   // Sweep out one row at a time
   for (int p = dim-1; p>=0; p--)
      {
      // Find the highest value to pivot around, in column p
      double bestval = 0.0;
      int bestrow = 0;
      for (int i=0; i<=p; i++)
         {
         double val=A[rows[i]][p];
         if (fabs(val) > bestval) 
            {
            bestval = fabs(val);
            bestrow = i;   // Actually rows[i]
            }
         }

      // Return early if the determinant is zero
      if (bestval == 0.0) return 0.0;

      // Now, select this row as a pivot.  Swap this row with row p
      if (bestrow != p)
         {
         int prow = rows[bestrow];
         rows[bestrow] = rows[p];   // Replace pivot row by last row
         rows[p] = prow;
         sign = -sign;		    // Keep track of sign 
         }

      // Clear out all the values above and to the right
      for (int i=0; i<p; i++)
         {
         int ii = rows[i];
         double fac = A[ii][p] / A[rows[p]][p];
         
         // Must do this to all the columns
         for (int j=0; j<dim; j++)
            A[ii][j] -= A[rows[p]][j] * fac;
         }
      }

   // Now compute the determinant
   double det = sign;
   for (int i=0; i<dim; i++)
      det *= A[rows[i]][i];
   return det;
   }

void do_scale (
	PolyMatrix &Q, 
	PolyDegree degree, 
        double &scale_factor,	// Value that x is multiplied by
        bool degree_by_row,	// Estimate degree from row degrees
        int dim			// Actual dimension of the matrix
	)
   {
   // Scale the variable so that coefficients of low and high order are equal
   // There is an assumption made here that the high order term of the
   // determinant can be computed from the high-order values of each term,
   // which is not in general true, but is so in the cases that we consider.

   // First step is to compute these values
   double low_order, high_order;
   int total_degree;

   // Find the coefficient of minimum degree term
   double A[Nrows][Nrows];
   for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
         A[i][j] = Q[i][j][0];

   low_order = quick_compute_determinant (A, dim);
   // printf ("Low order = %.7e\n", low_order);

   // Find the coefficient of maximum degree term
   total_degree = 0;
   for (int i=0; i<dim; i++)
      {
      // Find what the degree of this row is
      int rowdegree = -1;
      if (degree_by_row)
         {
         for (int j=0; j<dim; j++)
            if (degree[i][j] > rowdegree) rowdegree = degree[i][j];

         for (int j=0; j<dim; j++)
            if (degree[i][j] < rowdegree) A[i][j] = 0.0;
            else A[i][j] = Q[i][j][rowdegree];
         }
      else
         {
         for (int j=0; j<dim; j++)
            if (degree[j][i] > rowdegree) rowdegree = degree[j][i];

         for (int j=0; j<dim; j++)
            if (degree[j][i] < rowdegree) A[j][i] = 0.0;
            else A[j][i] = Q[j][i][rowdegree];
         }

      // Accumulate the row degree
      total_degree += rowdegree;
      }

   high_order = quick_compute_determinant (A, dim);
   // printf ("High order = %.7e\n", high_order);

   // Now, work out what the scale factor should be, and scale
   scale_factor = pow(fabs(low_order/high_order), 1.0 / total_degree);
   // printf ("Scale factor = %e\n", scale_factor);
   for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
         {
         double fac = scale_factor;
         for (int d=1; d<=degree[i][j]; d++)
            {
            Q[i][j][d] *= fac;
            fac *= scale_factor;
            }
         }
   }

void find_polynomial_determinant (
	PolyMatrix &Q, 
	PolyDegree deg, 
	int rows[Nrows], // This keeps the order of rows pivoted on. 
	int dim		// Actual dimension of the matrix
	)
   {
   // Compute the polynomial determinant - we work backwards from
   // the end of the matrix.  Do not bother with pivoting

   // Polynomial to start with
   double aa = 1.0;
   double *a00 = &aa;
   int deg00 = 0;

   // Initialize the list of rows
   for (int i=0; i<dim; i++)
      rows[i] = dim-1-i;

   // The row to pivot around.  At end of the loop, this will be 
   // the row containing the result.
   int piv;

   for (int p = dim-1; p>=1; p--)
      {
      // We want to find the element with the biggest high order term to
      // pivot around

#define DO_PARTIAL_PIVOT
#ifdef  DO_PARTIAL_PIVOT
      double bestval = 0.0;
      int bestrow = 0;
      for (int i=0; i<=p; i++)
         {
         double val=Q[rows[i]][p][deg[rows[i]][p]];
         if (fabs(val) > bestval) 
            {
            bestval = fabs(val);
            bestrow = i;   // Actually rows[i]
            }
         }

      // Now, select this row as a pivot.  Also keep track of rows pivoted
      piv = rows[bestrow];
      rows[bestrow] = rows[p];   // Replace pivot row by last row
      rows[p] = piv;
#else

      piv = rows[p];

#endif

// #define RH_DEBUG
#ifdef RH_DEBUG
#undef RH_DEBUG
      // Print out the pivot
      printf ("Pivot %d = \n", p);
      for (int i=0; i<=deg[piv][p]; i++)
         printf ("\t%16.5e\n", Q[piv][p][i]);
#endif

      // Set up a matrix for Toeplitz
      BMatrix B;
      int current_size = 0;

      // Also the Toeplitz vector
      double toep[Maxdegree+1];
      for (int i=0; i<=deg00; i++)
        {
        toep[i] = 0.0;
        for (int j=0; j+i<=deg00; j++)
           toep[i] += a00[j] * a00[j+i];
        }

      // Clear out all the values above and to the right
      for (int i=0; i<p; i++)
         {
         int iii = rows[i];
         for (int j=0; j<p; j++)
            cross_prod (
               Q[piv][p], deg[piv][p],
               Q[piv][j], deg[piv][j],
               Q[iii][p], deg[iii][p],
               Q[iii][j], deg[iii][j],
               a00, toep, deg00,
               Q[iii][j], deg[iii][j],	// Replace original value
               B, current_size
               );
         }

      // Now, update to the next
      a00 = &(Q[piv][p][0]);
      deg00 = deg[piv][p];
      }

   // Now, the polynomial in the position Q(0,0) is the solution
   }

//=========================================================================
//  The rest of this code is for stand-alone testing
//=========================================================================

#ifndef BUILD_MEX
#ifdef POLYDET_HAS_MAIN

void copy_poly (
      PolyMatrix pin, PolyDegree din, PolyMatrix pout, PolyDegree dout)
   {
   memcpy (pout, pin, sizeof(PolyMatrix));
   memcpy (dout, din, sizeof(PolyDegree));
   }

int accuracy_test_main (int argc, char *argv[])
   {
   // Try this out

   // To hold the matrix and its degrees
   PolyMatrix p;
   PolyDegree degrees;
   int pivotrows[Nrows];

   //--------------------------------------------------------

   // Generate some data
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         degrees[i][j] = ColumnDegree[j];

   // Now, fill out the polynomials
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         for (int k=0; k<=ColumnDegree[j]; k++)
            p[i][j][k] = urandom();
         for (int k=ColumnDegree[j]+1; k<=Maxdegree; k++)
            p[i][j][k] = 0.0;
         }

   //--------------------------------------------------------

   // Back up the matrix
   PolyMatrix pbak;
   PolyDegree degbak;
   copy_poly (p, degrees, pbak, degbak);

   //---------------------
   // Find determinant, then evaluate
   //---------------------

   // Now, compute the determinant
   copy_poly (pbak, degbak, p, degrees);

   // Preprocess
   double scale_factor = 1.0;
   det_preprocess_6pt (p, degrees, 3);
   do_scale (p, degrees, scale_factor, true);

   // Find the determinant
   find_polynomial_determinant (p, degrees, pivotrows);

   // Print out the solution
   const double print_solution = 0;
   if (print_solution)
      {
      printf ("Solution is\n");
      for (int i=0; i<=degrees[pivotrows[0]][0]; i++)
         printf ("\t%16.5e\n", p[pivotrows[0]][0][i]);
      }

   // Now, evaluate and print out
   double x = 1.0;
   eval_poly (p, degrees, x);

   double val1 = p[pivotrows[0]][0][0];

   //---------------------
   // Now, evaluate first
   //---------------------

   copy_poly (pbak, degbak, p, degrees);

   // Now, evaluate and print out
   eval_poly (p, degrees, x);
   find_polynomial_determinant (p, degrees, pivotrows);

   double val2 = p[pivotrows[0]][0][0];
   double diff = fabs((fabs(val1) - fabs(val2))) / fabs(val1);

   printf ("%18.9e\t%18.9e\t%10.3e\n", val1, val2, diff);

   return 0;
   }

int timing_test_main (int argc, char *argv[])
   {
   // Try this out

   // To hold the matrix and its degrees
   PolyMatrix p;
   PolyDegree degrees;
   int pivotrows[Nrows];

   //--------------------------------------------------------

   // Generate some data
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         degrees[i][j] = ColumnDegree[j];

   // Now, fill out the polynomials
   for (int i=0; i<Nrows; i++)
      for (int j=0; j<Ncols; j++)
         {
         for (int k=0; k<=ColumnDegree[j]; k++)
            p[i][j][k] = urandom();
         for (int k=ColumnDegree[j]+1; k<=Maxdegree; k++)
            p[i][j][k] = 0.0;
         }

   //--------------------------------------------------------

   // Back up the matrix
   PolyMatrix pbak;
   PolyDegree degbak;
   copy_poly (p, degrees, pbak, degbak);

   // Now, compute the determinant
   for (int rep=0; rep<NRepetitions; rep++)
      {
      copy_poly (pbak, degbak, p, degrees);
      find_polynomial_determinant (p, degrees, pivotrows);
      }

   return 0;
   }

//===========================================================================

int main (int argc, char *argv[])
   {
   // Now, compute the determinant
   // for (int rep=0; rep<NRepetitions; rep++)
   //  accuracy_test_main (argc, argv);

   timing_test_main (argc, argv);

   return 0;
   }

#endif
#endif  // BUILD_MEX

