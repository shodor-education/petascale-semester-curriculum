/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 8: OpenACC
 * Lesson 3: N-Body Mechanics in OpenACC
 * File: N-body-OpenACC1.cpp
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

#include <stdlib.h>     // srand, rand
#include <math.h>       // sqrt, exp, pow, abs
#include <iostream>     // cout
#include <iomanip>      // setw
#include <sys/time.h>        // clock
#include <fstream>      // fout
#include <omp.h>          // omp_get_thread_num()
using namespace std;



#define index(row, col, ncols) (row*ncols + col)

void acceleration(double x[], double y[], double z[], int itime, int Nbodies, double mass, double ax[], double ay[], double az[]); 

int main() {

   int Nbodies=10000;                   // Number of Bodies
   int Ntime=100;                     // Number of time steps
   float LBox=1.4372e14;               // 2*LBox=Size of the Box in meters
   //int Nbodies=2;                    // Number of Bodies
   //float LBox=1.4372e12;               // 2*LBox=Size of the Box in meters
   double mass=2e30/Nbodies;           // The size of each mass
   // double Tlength=200.0*365.25*24*3600;// 200 years in seconds
   double Tlength=.5*365.25*24*3600;// 200 years in seconds
   double dt=Tlength/Ntime;            // size of time step
   double scalelength=50;       // the scale length of the expotnential decay of the initial mass distribution
                                // A larger scalelength gives a faster decay and thus tighter distribution
			        // It will also take longer to populate initial conditions
   double omega=0.1*2*M_PI/Tlength;     // The ratio of KE to abs(PE) for each particle initially
			
   double *x,*y,*z;                  // For storing x, y, & z positions as a function of time
   double *ax,*ay,*az;               // For ax, ay, & az accelerations as a function of time
   double tempx,tempy,tempz,tempr;   // For storing temporary values
   double prob1, prob2;              // For storing probabilities for accept/reject algorythm
   double vx, vy;                // For storing temporary x and y velocities for the initial Euler step 
   int itime;        // time and body indicies
   int itimep1, itimem1;             // extra time indicices for verlet loop
   struct timeval t1,t2,t3,t4,t5; // Various clock points
   

   x=new double[Ntime*Nbodies]; // This vector stores our x coordinates for each body and each time step
   y=new double[Ntime*Nbodies]; // This vector stores our y coordinates for each body and each time step
   z=new double[Ntime*Nbodies]; // This vector stores our z coordinates for each body and each time step
   ax=new double[Nbodies];      // This vector stores our ax acceleration for each body at a single time step
   ay=new double[Nbodies];      // This vector stores our ay acceleration for each body at a single time step
   az=new double[Nbodies];      // This vector stores our az acceleration for each body at a single time step

   // We can access a particular coordinate at a particular time step with x[index(itime,ibody,Nbodies)]
   //    where itime is the time index and ibody is the body number index
   
   // The first step is we need to initialize the initial locations of each body 

   gettimeofday(&t1,NULL);
   itime=0;
   // In order to paralize we have to change rand to rand_r
   // rand_r is threadsafe!  To keep each processor from
   // generating masses at the same points, we also need to
   // make sure each processor has a different random seed
   // Must use a parallel and then a parallel for because
   // we need the thread number to seed the generator, which 
   // should be outside the loop!
 
   {
   unsigned int seed=t1.tv_usec;
   for (int ibody1=0; ibody1<Nbodies; ibody1++) {
           prob1=0.0;
	   prob2=1.0;
	   while (prob2>=prob1) {
	           tempx=rand_r(&seed); // Recall rand returns an integer between 0 and RAND_MAX
	           tempx=((tempx/RAND_MAX)-0.5)*2*LBox; // Now tempx holds a number between -LBox and + LBox
	           tempy=rand_r(&seed);
	           tempy=((tempy/RAND_MAX)-0.5)*2*LBox;
	           tempz=rand_r(&seed);
	           tempz=((tempz/RAND_MAX)-0.5)*2*LBox;
                   // The above code will make our masses scattered evenly throughout the box.
	           // We really want the mass to be more concentrated at the center, so we are going to accept
	           // or reject with a probability of exp(-scalelength*r^2/LBox)
		   // This code +  the while loop accomplishes this
                   tempr=sqrt(tempx*tempx+tempy*tempy+tempz*tempz);
                   prob1=exp(-scalelength*tempr/LBox);
                   prob2=rand_r(&seed);
                   prob2=(prob2/RAND_MAX); }
            x[index(itime,ibody1,Nbodies)]=tempx;
            y[index(itime,ibody1,Nbodies)]=tempy;
            z[index(itime,ibody1,Nbodies)]=tempz;}
   }

   gettimeofday(&t2,NULL);
   tempr=(t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
   cout << "Time to init: " << tempr << " ms" << endl;
     
   // Now we need to calculate the possiton at the second time point itime=1
   // For this calculations we will need accelerations and initial velocities
   // and the expression x(1)=x(0)+v(0)*dt+0.5*a*dt*dt (we are assuming a constant a over the interval)
   // for initial velocities, we will make the system rotate with a constant angular velocity omega, thus
   // vx=-omega*y[index(itime,ibody1,Nbodies)];
   // vy=omega*x[index(itime,ibody1,Nbodies)]; 

   // Before we use those, we need to calculate the accelerations
   acceleration(x,y,z,itime,Nbodies,mass,ax,ay,az);

   // We will set the Kinetic energy of each partical to be KEfrac*abs(pot[ibody]) 
   // KE=(0.5)*mass*v^2 ===> v=sqrt(2*KE/mass)=sqrt(2*KEfrac*abs(pot[ibody]))
   // vx=-v*sin(theta)=-v*y/r, where r is the distance from the z axis
   // vy=v*cos(theta)=v*x/r
   vx=0;
   vy=0;
   itimep1=itime+1;
   for (int ibody1=0; ibody1<Nbodies; ibody1++) {
	   vx=-omega*y[index(itime,ibody1,Nbodies)];  // pot is always negative, thus we drop the - and the abs
	   vy=omega*x[index(itime,ibody1,Nbodies)];   // pot is always negative, thus we add the - and drop the abs
           x[index(itimep1,ibody1,Nbodies)]=x[index(itime,ibody1,Nbodies)]+vx*dt+0.5*ax[ibody1]*dt*dt;
           y[index(itimep1,ibody1,Nbodies)]=y[index(itime,ibody1,Nbodies)]+vy*dt+0.5*ay[ibody1]*dt*dt;
           z[index(itimep1,ibody1,Nbodies)]=z[index(itime,ibody1,Nbodies)]+0.5*az[ibody1]*dt*dt;} // vz=0

   gettimeofday(&t3,NULL);
   tempr=(t3.tv_sec - t2.tv_sec) * 1e6 + (t3.tv_usec - t2.tv_usec);
   cout << "Time for first step: " << tempr << " ms" << endl;

   // Now the meat!  Verlet integration
   // x(i)=2*x(i-1)-x(i-2)+a(i-1)*dt*dt

   for(itime=1;itime<Ntime-1;itime++){
	   acceleration(x,y,z,itime,Nbodies,mass,ax,ay,az);
	   itimep1=itime+1;
	   itimem1=itime-1;
	   for(int ibody1=0;ibody1<Nbodies;ibody1++){
                   x[index(itimep1,ibody1,Nbodies)]=2*x[index(itime,ibody1,Nbodies)]-x[index(itimem1,ibody1,Nbodies)]+ax[ibody1]*dt*dt;
                   y[index(itimep1,ibody1,Nbodies)]=2*y[index(itime,ibody1,Nbodies)]-y[index(itimem1,ibody1,Nbodies)]+ay[ibody1]*dt*dt;
                   z[index(itimep1,ibody1,Nbodies)]=2*z[index(itime,ibody1,Nbodies)]-z[index(itimem1,ibody1,Nbodies)]+az[ibody1]*dt*dt;}}

   // cout << "Verlet Complete" << endl;
   gettimeofday(&t4,NULL);
   tempr=(t4.tv_sec - t3.tv_sec) * 1e6 + (t4.tv_usec - t3.tv_usec);
   cout << "Time to run Verlet: " << tempr << " ms" << endl;

   FILE *filex, *filey, *filez;
   filex=fopen("xout.bin","wb");
   filey=fopen("yout.bin","wb");
   filez=fopen("zout.bin","wb");
   for(itime=0; itime<Ntime; itime++) {
		   fwrite(&x[itime*Nbodies],sizeof(double),Nbodies,filex);
		   fwrite(&y[itime*Nbodies],sizeof(double),Nbodies,filey);
		   fwrite(&z[itime*Nbodies],sizeof(double),Nbodies,filez);}
   fclose(filex);
   fclose(filey);
   fclose(filez);

   ofstream myfile;
   myfile.open ("out3.txt");

   for(itime=0; itime<Ntime; itime++) {
           for(int ibody1=0; ibody1<Nbodies; ibody1++) {
	           myfile << setw(15) << x[index(itime,ibody1,Nbodies)]; 
		   myfile << setw(15) << y[index(itime,ibody1,Nbodies)]; 
		   myfile << setw(15) << z[index(itime,ibody1,Nbodies)] <<endl;}}

   myfile.close();

   gettimeofday(&t5,NULL);
   tempr=(t5.tv_sec - t4.tv_sec) * 1e6 + (t5.tv_usec - t4.tv_usec);
   cout << "Time for IO: " << tempr << " ms" << endl;
   tempr=(t5.tv_sec - t1.tv_sec) * 1e6 + (t5.tv_usec - t1.tv_usec);
   cout << "Total Time: " << tempr << " ms" << endl;
   
   return 0;
}

void acceleration(double *restrict x, double *restrict y, double *restrict z, int itime, int Nbodies, double mass, double *restrict ax, double *restrict ay, double *restrict az) {

        double F; // temporary to hold force/(m*r)
	double dx, dy, dz, r; // tempories to hold the components of the displacement vector and the distance between the objects
	int ibody2; // inner loop variable
        double G=6.67430e-11;    // Universal Gravitational constant in m^3 kg^-1 s^-2
                                 // https://physics.nist.gov/cgi-bin/cuu/Value?bg

	#pragma acc kernels
	for(int ibody1=0;ibody1<Nbodies;ibody1++) {
		dx=x[index(itime,ibody1,Nbodies)];
		dy=y[index(itime,ibody1,Nbodies)];
		dz=z[index(itime,ibody1,Nbodies)];
                r=sqrt(dx*dx+dy*dy+dz*dz)+1.0e7; // 1.0e7 is to soften the potential, prevent "explosions"
		F=(G*10*mass*Nbodies/(r*r*r)); // a central mass/star/something with the 10x mass as the system
		ax[ibody1]=-F*dx;
		ay[ibody1]=-F*dy;
		az[ibody1]=-F*dz;
		for (ibody2=0;ibody2<ibody1-1;ibody2++) {
		        dx=x[index(itime,ibody1,Nbodies)]-x[index(itime,ibody2,Nbodies)];
		        dy=y[index(itime,ibody1,Nbodies)]-y[index(itime,ibody2,Nbodies)];
		        dz=z[index(itime,ibody1,Nbodies)]-z[index(itime,ibody2,Nbodies)];
                        r=sqrt(dx*dx+dy*dy+dz*dz)+1.0e7; // 1.0e7 is to soften the potential, prevent "explosions"
			F=(G*mass/(r*r*r)); // Only 1 mass because we are dividing theough by the mass of the particle
			ax[ibody1]-=F*dx;
			ay[ibody1]-=F*dy;
			az[ibody1]-=F*dz;}
                // We don't do ibody1=ibody2 because masses don't pull on themselves
		for (ibody2=ibody1+1;ibody2<Nbodies;ibody2++) {
		        dx=x[index(itime,ibody1,Nbodies)]-x[index(itime,ibody2,Nbodies)];
		        dy=y[index(itime,ibody1,Nbodies)]-y[index(itime,ibody2,Nbodies)];
		        dz=z[index(itime,ibody1,Nbodies)]-z[index(itime,ibody2,Nbodies)];
                        r=sqrt(dx*dx+dy*dy+dz*dz)+1.0e7; // 1.0e7 is to soften the potential, prevent "explosions"
			F=(G*mass/(r*r*r)); // Only 1 mass because we are dividing theough by the mass of the particle
			ax[ibody1]-=F*dx;
			ay[ibody1]-=F*dy;
			az[ibody1]-=F*dz;}}}
