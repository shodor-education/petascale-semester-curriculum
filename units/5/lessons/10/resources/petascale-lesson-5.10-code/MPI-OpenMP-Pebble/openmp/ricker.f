        program wavelet
! Blue Waters Petascale Semester Curriculum v1.0
! File: MPI-OpenMP-Pebble/openmp/ricker.f
! Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
! Included in the following lessons:
! - Unit 5 (MPI) Lesson 10: Wave Propagation in MPI
! - Unit 6 (Hybrid MPI + OpenMP) Lesson 3: Pebble in Pond Wave Equation
! - Unit 8 (OpenACC) Lesson 1: Accelerating Scientific Applications
! - Unit 10 (Productivity and Visualization) Lesson 3: Visualization 1
!
! Copyright (c) 2020 The Shodor Education Foundation, Inc.
!
! Browse and search the full curriculum at
! <http://shodor.org/petascale/materials/semester-curriculum>.
!
! We welcome your improvements! You can submit your proposed changes to this
! material and the rest of the curriculum in our GitHub repository at
! <https://github.com/shodor-education/petascale-semester-curriculum>.
!
! We want to hear from you! Please let us know your experiences using this
! material by sending email to petascale@shodor.org
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as published
! by the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Affero General Public License for more details.
!
! You should have received a copy of the GNU Affero General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.

        integer*4  npoints
        real*4     ricker(10000)
        real*4     frequency
        real*4     dt

        npoints = 400
         
        frequency = 30.0e00
        dt = 0.0025e00
        call source(npoints,ricker,frequency,dt)
        
        end
        subroutine source(npoints,ricker,frequency,dt)
!
!       create a Ricker wavelet with a center "frequency".
!
!       output is a source vector - array(npoints)
!

        integer*4  npoints
        real*4     ricker(*)
        real*4     frequency
        real*4     dt

!     zero entire array
        do i=1,npoints
         ricker(i) = 0.0e00
        enddo

        pi = 3.1415926e00
        cpift = (pi*frequency)**2.0e00

        time = -48.0e00*dt
        do it =1,100
        ricker(it) = (1.0e00-2.0e00*cpift*time**2)*exp(-cpift*time**2)
        time = time + dt
        write(6,*) it,time,ricker(it)
        enddo

        return
        end

        
