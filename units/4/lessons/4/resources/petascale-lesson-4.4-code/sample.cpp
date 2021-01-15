/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 4: OpenMP Target Offload
 * File: sample.c
 * Developed by Maria Pantoja for the Shodor Education Foundation, Inc.
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

#include <set>
#include <iostream>
#include <assert.h>
#include <vector>

using namespace std ;

void vec_mult (int N, const int n_print =10){
	double p[N] , v1[N] , v2[N] ;
	for(int ii =0; ii <N; ++ ii ) {
		p [ ii ] = ii %5;
		v1 [ ii ] = ii %6;
		v2 [ ii ] = ii %7;
	}

	int i ;
	#pragma omp target map( to : v1[0:N] , v2[0:N] , p[0:N] )
	#pragma omp parallel for private(i)
	for ( i =0; i <N; i ++ ) {
		p [i] = v1 [i] * v2 [i] ;
	}

	int num_print = 0;
	if( n_print > N) num_print = N;
	else num_print = n_print ;
	for(int ii =0; ii <num_print ; ++ ii ) 
		cout << p [ ii ] << "  " ;
	cout << endl ;
}

int main (int argc , char * * argv )
{
	cout << "#########################################" << endl ;
	cout << "#######      Test Start         #########" << endl ;
	cout << "#########################################" << endl << endl ;
	vec_mult (100000 );
	cout << "Test complete!\n" ;
	return 0;
}
