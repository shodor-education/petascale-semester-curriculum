/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 8: OpenACC
 * Lesson 4: Solving the Heat Equation via Jacobi's Method
 * File: Heat-OpenACC1.cpp
 * Developed by Justin Oelgoetz for the Shodor Education Foundation, Inc.
 *
 * Copyright (c) 2020 The Shodor Education Foundation, Inc.
 *
 * Browse and search the full curriculum at
 * <http://shodor.org/petascale/materials/semester-curriculum>.
 *
 * We welcome your improvements! You can submit your proposed changes to this
 * material and the rest of the curriculum in our GitHub repository at
 * <https://github.com/shodor-education/petascale-semester-curriculum>.
 *
 * We want to hear from you! Please let us know your experiences using this
 * material by sending email to petascale@shodor.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <math.h>       // sqrt, exp, pow, abs
#include <iostream>     // cout
#include <iomanip>      // setw
#include <sys/time.h>        // clock
#include <fstream>      // fout
#include <omp.h>          // omp_get_thread_num()
using namespace std;



int main() {
	int NMesh=750;
	// These are used to prevent an infinite loop
	int iter;
	int MaxIter=10000;
	int index;

	// Columns of integers to store the rows and columns of non-zero, off diagonal elements
	// Declaring pointers with restrict allows for memory optimization and offloading
	// It states that these pointers will point to distinct memory locations
	int *restrict Arow, *restrict Acol;
	// Values of non-zero elements
	double *restrict Avalue;
	// Values of diagonal elements
	double *restrict diag;
	// Values of the right side of the equation
	double *restrict b;
	// used for storing the solution and intermediate steps
	double *restrict interim;
	double *restrict x;
	// Number of non zero elements
	int Nnz;
	// The maximum temperature difference between iterations
	double conv;
	// A temporary variable
	double temp;
	// The convergence criteria (how good of an answer we want)
	double convcriteria=0.001;
	/* For debugging
	double *A;
	A=new double[NMesh*NMesh*NMesh*NMesh]; */

	double spenttime; // For holding elapsed time calculations
	struct timeval t1,t2,t3,t4; // Various clock points

	diag=new double[NMesh*NMesh]; // this is dense
	b=new double[NMesh*NMesh];    // this is dense
	x=new double[NMesh*NMesh];  // this is dense
	interim=new double[NMesh*NMesh];  // this is dense

	Nnz=NMesh*NMesh*4; // This number is actually slightly larger than what what
	             // we will need.  
	Arow=new int[NMesh*NMesh+1];
	Acol=new int[Nnz];
	Avalue=new double[Nnz];

        gettimeofday(&t1,NULL);


	index=-1; // Where in the off diagonal array we are placing the value
	for (int i=0;i<NMesh*NMesh;i++) {
		Arow[i]=index+1;
		x[i]=0;
		if (i<NMesh) {
			diag[i]=1;
			b[i]=0;}
		else if ((NMesh*NMesh-i)<NMesh) {
			diag[i]=1;
			b[i]=0;}
		else if ((i%NMesh)==0) {
			diag[i]=1;
			b[i]=0;}
		else if ((i%NMesh)==(NMesh-1)) {
			diag[i]=1;
			b[i]=0;}
		else if (i==(NMesh*NMesh/2+NMesh/2)) {
			diag[i]=1;
			b[i]=100;
		        x[i]=100;}
		else {
			diag[i]=-4;
			index=index+1;
			b[i]=0;
			Acol[index]=i-NMesh;
			Avalue[index]=1;
			index=index+1;
			Acol[index]=i-1;
			Avalue[index]=1;
			index=index+1;
			Acol[index]=i+1;
			Avalue[index]=1;
			index=index+1;
			Acol[index]=i+NMesh;
			Avalue[index]=1;} }
	Nnz=index+1;


/* This section is for debugging
	for (int i=0;i<NMesh*NMesh*NMesh*NMesh;i++) {
		A[i]=0.0;}
        for (int i=0;i<NMesh*NMesh;i++) {
                 for (int j=Arow[i];j<Arow[i+1];j++){
                        A[i*NMesh*NMesh+Acol[j]]=Avalue[j];}}
        FILE *aout;
        aout=fopen("aout.bin","wb");
        fwrite(A,sizeof(double),NMesh*NMesh*NMesh*NMesh,aout);
        fclose(aout);
        FILE *dout;
        dout=fopen("dout.bin","wb");
        fwrite(diag,sizeof(double),NMesh*NMesh,dout);
        fclose(dout);
        FILE *bout;
        bout=fopen("bout.bin","wb");
        fwrite(b,sizeof(double),NMesh*NMesh,bout);
        fclose(bout);
*/

        gettimeofday(&t2,NULL);
        spenttime=(t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
        cout << "Matrix Loaded: " << spenttime << " ms" << endl;

	conv=100; // this is the maximum temperature, thus by definition not converged
	iter=0;
	int k;
	int j;
#pragma acc data copyin(Arow[NMesh*NMesh+1],Acol[NMesh*NMesh*4],Avalue[NMesh*NMesh*4]) copy(x[NMesh*NMesh]) create(interim[NMesh*NMesh])
        while ((conv>convcriteria)&&(iter<MaxIter)) {
		iter=iter+1;
#pragma acc parallel loop
		for (int i=0;i<NMesh*NMesh;i++) {
		        interim[i]=0.0;}
#pragma acc parallel loop private(k,j) present(Arow[NMesh*NMesh+1],Acol[NMesh*NMesh*4],Avalue[NMesh*NMesh*4])
		for (int i=0;i<NMesh*NMesh;i++) {
			for (j=Arow[i];j<Arow[i+1];j++){
				k=Acol[j];
	         	        interim[i]=interim[i]+Avalue[j]*x[k];}}
		conv=0.0;
#pragma acc parallel loop private(temp) reduction(+:conv)
		for (int i=0;i<NMesh*NMesh;i++) {
			temp=(b[i]-interim[i])/diag[i];
			conv=conv+fabs(x[i]-temp);
		        x[i]=temp;}}
	
        gettimeofday(&t3,NULL);
        spenttime=(t3.tv_sec - t2.tv_sec) * 1e6 + (t3.tv_usec - t2.tv_usec);
        cout << "Time to solve: " << spenttime << " ms" << endl;

        FILE *uout;
        uout=fopen("uout.bin","wb");
        fwrite(x,sizeof(double),NMesh*NMesh,uout);
        fclose(uout);

        gettimeofday(&t4,NULL);
        spenttime=(t4.tv_sec - t3.tv_sec) * 1e6 + (t4.tv_usec - t3.tv_usec);
        cout << "Time for IO: " << spenttime << " ms" << endl;
        spenttime=(t4.tv_sec - t1.tv_sec) * 1e6 + (t4.tv_usec - t1.tv_usec);
        cout << "Total Time: " << spenttime << " ms" << endl;	

}

