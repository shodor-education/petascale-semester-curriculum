          program mmatrix
! Blue Waters Petascale Semester Curriculum v1.0
! File: mmatMultiply/mmatrix.f
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


!       matrix multipy test code for scaling and timing studies.
!      

          parameter (ndx=1024,ndz=1024)

          real*4  am(ndx,ndz)
          real*4  bm(ndx,ndz)
          real*4  cm(ndx,ndz)

!
!      fill arrays am and bm with random numbers and 
!      then multiply am by bm to get cm
!
!     repeat 10 times to get better averages
!
       do itimes=1,10
!
          do kz=1,ndz
           do ix=1,ndx
            am(ix,kz) = 1000.0e00*rand() -500.0e00
            bm(ix,kz) = 1000.0e00*rand() -500.0e00
            cm(ix,kz) = 0.0e00
           enddo
          enddo
!
!
!
!
          do kz=1,ndz

           do ix=1,ndx
            do jy=1,ndx
             cm(ix,kz) = cm(ix,kz) + am(ix,jy)*bm(jy,kz)
            enddo

           enddo
          enddo
!  repeat loop for better times
         enddo

         call exit()
         end
