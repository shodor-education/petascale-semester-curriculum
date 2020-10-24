/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 4: OpenMP
 * Lesson 7: OpenMP Applications & Practice
 * File: integration_omp.c
 * Developed by Widodo Samyono for the Shodor Education Foundation, Inc.
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

float f(float x) {
return (x*x);
}
int main() {
     int i, SECTIONS = 1000;
     float height = 0.0;
     float area = 0.0, y = 0.0, x = 0.0;
     float dx = 1.0/(float)SECTIONS;

/* Start of the parallel section */
     #pragma omp parallel for private(x, y) reduction(+: area)
           for( i = 0; i < SECTIONS; i++){
                x = i*dx;
                y = f(x);
                area += y*dx;
              }
 /* End of  the parallel section */
  printf("Area under the curve is %f\n",area);
  return (0);
}

