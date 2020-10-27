          Program  mframe
! Blue Waters Petascale Semester Curriculum v1.0
! File: testGraphicsMagick/mframe.f
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
!      Make a simple rectangular movie frame
!
!
        Parameter (ndx=1024,ndz=512)

        real*4  mframe_float(ndx,ndz)

        integer*4 mframe_int  (ndx,ndz)
        integer*4 mframe_pixel(ndx,ndz)

        character*24  filename
        character*18  dirfilename

!
!      fill mframe_floats full of random numbers (-1.0 to +1.0)
!
!      the Fortran random number generator range is 0.0 -> 1.0
!
       do kz=1,ndz
        do ix=1,ndx
         mframe_float(ix,kz) = rand()*2.0e00-1.0e00
        enddo
       enddo
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
     
       write(6,*) "   Random array is Ndx by Ndz in size "
       write(6,*) "   Array Ndx =",Ndx 
       write(6,*) "   Array Ndz =",Ndz 
       write(6,*) "                   "
       write(6,*) "   Array maximum positive value is: ",dmaxp
       write(6,*) "   Array maximum negative value is: ",dminn
       write(6,*) "                   "
!
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

!
!
!       a bit tricky now we write a Fortran random access file...
!
!
        nrecl = 4*ndx
!
!      records are in bytes
!
        filename(1:24) ="data/gray_frame00000.dat"
        dirfilename(1:18) ="../testImageMagik/"


!
!   write 10 frames
!
      kframe = 10

      kzline = ndz/20
!
      do ifmm =1,kframe 
!
!     modify each frame with a solid black line
!
        do ix=1,ndx
         mframe_pixel (ix,kzline) =  none  
        enddo

      kzline = kzline + 16

      irec = 1
        write(filename(16:20),200) ifmm-1
        open(10,file=filename(1:24),form="unformatted",
     x     access="direct",recl=nrecl)
!

       do kz=1,ndz
        write(10,rec=irec) (mframe_pixel(ix,kz),ix=1,ndx)
        irec = irec + 1
       enddo

       close(10)

      enddo


       kframechars = 24
!
!      Build GraphicsMagick header file and file name file.
!
!      Directory base file name -->  filename(1:15) ="data/gray_frame"
!
          open(9,file="data/dirfilename",form="formatted",
     x           recl=18,
     x           access="direct")
 244      format(18a1)
          write(9,244,rec=1)  (dirfilename(k:k),k=1,18)
          close(9)
!
!       write hdr file
!
          open(9,file="data/gray_frame.hdr",form="unformatted",
     x           recl=16,
     x           access="direct")
          write(9,rec=1)  kframe,ndx,ndz,kframechars
          close(9)
!
!       filename character data needs a formatted file
!
          open(9,file="data/gray_frame.filenames",form="formatted",
     x           recl=24,
     x           access="direct")
          filename(1:24) = "data/gray_frame00000.dat"
          irecl = 1
          do ifn=1,kframe
           write(filename(16:20),200) ifn-1
 200       format(i5.5)
           write(9,222,rec=irecl)  (filename(k:k),k=1,24)
 222       format(24a1)
           irecl=irecl + 1
          enddo
          close(9)

        
       write(6,*) "             "
       write(6,*) " Move to the mimage directory and run --> tframe"
       write(6,*) "             "
!
!      ok we should have an GraphicsMagick file we can see

           write(6,218)  (dirfilename(k:k),k=1,18)
 218       format(" directory for data is: ",18a1)
!
!      with the tframe.c code in the mimage directory
!
       end
