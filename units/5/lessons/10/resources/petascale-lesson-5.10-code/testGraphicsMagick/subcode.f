          subroutine  float_pixel(frame_float,frame_pixel)
! Blue Waters Petascale Semester Curriculum v1.0
! File: testGraphicsMagick/subcode.f
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

!
!      Convert float to pixel a simple rectangular movie frame
!
!
!      Bad programming -- array size is fixed
!
!
        Parameter (ndx=512,ndz=512)

        real*4    frame_float(ndx,ndz)

        integer*4 frame_int  (ndx,ndz)
        integer*4 frame_pixel(ndx,ndz)

!
!      now find the maximum positive value in the float array
!      now find the maximum negative value in the float array
!
       dmaxp = 0.0e00
       dminn = 0.0e00

       do kz=1,ndz
        do ix=1,ndx
         if(mframe_float(ix,kz) .ge. dmaxp) dmaxp = mframe_float(ix,kz)
         if(mframe_float(ix,kz) .le. dminn) dminn = mframe_float(ix,kz)
        enddo
       enddo
     
!
!       now map -1.0   ->  0.0   ->  +1.0
!       to        0       128         255
!
!               float + 1.0        --> 0.0 -->  2.0
!              (float + 1.0)*0.5   --> 0.0 -->  1.0
!    int(255.0*(float + 1.0)*0.5)  -->   0 -->  255
!          
!
!    now build the integer array
!
       do kz=1,ndz
        do ix=1,ndx
!    integer to float happens across the = sign
!                   mfm = (mframe_float(ix,kz)+1.0e00)*127.5e00 
!        write(6,*) mfm,(mframe_float(ix,kz)+1.0e00)*127.5e00 

         mframe_int(ix,kz) = (mframe_float(ix,kz)+1.0e00)*127.5e00 
       
        enddo
       enddo

!    now convert the integer array into a ARGB pixel array
!
!    Alpha is opacity and is set to none - you will see all of this image
!                                   none = 255
!
!                    Alpha, Red, Green, and Blue
!
!   Hex values          ff   00     00        00   ==>  ff000000   Black
!   Hex values          ff   01     01        01   ==>  ff010101   Dark Grey
!   Hex values          ff   10     10        10   ==>  ff101010   Light Grey
!   Hex values          ff   ff     ff        ff   ==>  ffffffff   White
!
       none = -256*256*256
!
!    using 2's complement arithmetic we set the initial hex ff
!
       
       do kz=1,ndz
        do ix=1,ndx
         irgb = mframe_int(ix,kz)
         mframe_pixel (ix,kz) =  none + 256*256*irgb  + 256*irgb + irgb 
        enddo
       enddo

       return
       end
