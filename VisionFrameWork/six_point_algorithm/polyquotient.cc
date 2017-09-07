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

#include <string.h>
#include <stdio.h>
#include "hidden6.h"

const int NRepetitions = 100000;

static void toeplitz_get_b_vector (
	BMatrix B, 
	double *t, int st,	
	int &current_size, int required_size
	)
   {
   // Incrementally computes the matrix of back-vectors used in the Levinson
   // algorithm. The back-vectors bn are stored as rows in B.

   // Initialize B
   if (current_size <= 0)
      {
      B[0][0] = 1.0/t[0];
      current_size = 1;
      }

   // Build up the back vectors one by one
   for (int n=current_size; n<required_size; n++)
      {
      // Fill out row n of the matrix B

      // Compute out each vector at once
      double e = 0.0;
      for (int i=1; i<st && i<=n; i++)
         e += t[i] * B[n-1][i-1];

      // Write the next row of the matrix
      double cb = 1.0 / (1.0 - e*e);
      double cf = -(e*cb);

      // Addresses into the arrays for the addition
      double *b0 = &B[n-1][0];
      double *f0 = &B[n-1][n];
      double *bn = &(B[n][0]);

      // First term does not include b, last does not have f
      *(bn++) = *(--f0) * cf;
      while (f0 != &B[n-1][0])
         *(bn++) = *(b0++) * cb + *(--f0) * cf;
      *bn = *b0 * cb;
      }

   // Update the current dimension
   current_size = required_size;
   }

void polyquotient (
	double *a, int sa, 
	double *b, double *t, int sb, 
	double *q, int &sq,
	BMatrix B, int &current_size
	)
   {
   // Computes the quotient of one polynomial with respect to another
   // in a least-squares sense, using Toeplitz matrices
   
   // First, get the sizes of the vectors
   sq = sa - sb + 1;  // Degree of the quotient

   // Next get the back-vectors for the Levinson algorithm
   if (sq > current_size)
      toeplitz_get_b_vector (B, t, sb, current_size, sq);

#ifdef RH_DEBUG
   for (int i=0; i<sq; i++)
      {
      for (int j=0; j<sq; j++)
          printf ("%9.3f ", B[i][j]);
      printf ("\n");
      }
#endif

   // Initially no values
   memset(q, 0, sq*sizeof(double));

   // Next, compute the quotient, one at a time
   for (int n=0; n<sq; n++)
      {
      // Inner product of a and b
      double yn = 0.0;
      for (int i=0; i<sb; i++)
         yn += b[i] * a[i+n];

      // The error value
      double e = 0.0;
      for (int i=1; i<sb && i<=n; i++)
         e += t[i] * q[n-i];

#ifdef RH_DEBUG
      printf ("yn = %12.6f, e = %12.6f\n", yn, e);
#endif

      // Now, update the value of q
      double fac = yn - e;
      q[n] = 0.0;
      for (int i=0; i<=n; i++)
         q[i] += fac * B[n][i];
      }
   }

#ifndef BUILD_MEX
#ifdef POLYQUOTIENT_HAS_MAIN

int main (int argc, char *argv[])
   {
   // Try the thing out
   const int qsize = 21;
   const int bsize = 21;
   const int asize = qsize+bsize-1;

   int da = asize;
   int db = bsize;
   int dq = qsize;

   double a[asize];
   double b[bsize];
   double q[qsize];
   double t[bsize];

   // Fill out the polynomials with random values
   for (int i=0; i<dq; i++) q[i] = (double) i+1;
   for (int i=0; i<db; i++) b[i] = (double) i+1;

   // The matrix of back vectors
   BMatrix B;

   // Now, try the thing
   for (int row=0; row<9; row++)
      {
      // Do the test for pivoting on row "row"
      printf ("Pivoting on row %d\n", row);
      fflush(stdout);

      db = 2*row+1;
      da = 4*row+5;
      dq = 2*row+5;
      int reps = (9-row)*(9-row);

      // Multiply out to get a
      for (int i=0; i<da; i++) a[i] = 0.0;
      for (int i=0; i<db; i++)
         for (int j=0; j<dq; j++)
            a[i+j] += b[i]*q[j];

      // Also, multiply out to get the toeplitz vector
      for (int i=0; i<db; i++)
         {
         t[i] = 0.0;
         for (int j=0; j+i<db; j++)
            t[i] += b[j] * b[j+i];
         }

#ifdef RH_DEBUG
      printf ("a = \n");
      for (int i=0; i<da; i++) printf ("%7.2f\n", a[i]);

      printf ("b = \n");
      for (int i=0; i<db; i++) printf ("%7.2f\n", b[i]);

      printf ("q = \n");
      for (int i=0; i<dq; i++) printf ("%7.2f\n", q[i]);

      printf ("t = \n");
      for (int i=0; i<db; i++) printf ("%7.2f\n", t[i]);
#endif

      for (int rep=0; rep<NRepetitions; rep++)
         {
         int current_size = 0;
         for (int m=0; m<reps; m++)
            {
            polyquotient (a, da, b, t, db, q, dq, B, current_size);
            }
         }
      }

   printf ("Finished\n");
   fflush(stdout);

#ifdef RH_DEBUG
   // Now, print out the result
   for (int i=0; i<dq; i++)
      printf("%9.3f\n", q[i]);
#endif

   return 0;
   } 
   

#endif
#endif	// BUILD_MEX
