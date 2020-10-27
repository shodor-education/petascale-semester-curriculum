        Program design
! Blue Waters Petascale Semester Curriculum v1.0
! Unit 5: MPI
! Lesson 10: Wave Propagation in MPI
! File: MPI-Waves/design.f
! Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
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
!
!       Serial test code for the decompostion of one dimension
!       MPI test codes are based on running with nproc processors 
!
!       If the memory allocation of an array is a ndim
!
!       What happens if ndim is not an EXACT multiple nproc?
!       
!       Then the decomposition of ndim by nproc has a remainder.
!
!       Examples.................................
!
!              Ndim       Nproc      Variables       Remainder
!                                    per Processor
!              1000        10          100              0
!              1024        10          102              4
!
!                 What to do with this remainder??
!
!       Assign to some processor
!
!         rank   0    1    2    3    4    5    6    7    8    9
!
!     var/proc  10   10   10   10   10   10   10   10   10   14    lumped at the end
!     var/proc  10   10   10   10   10   10   11   11   11   11    even over last 4....
!
!
        integer*4   Nz,NumProc
        integer*4   Nzp
        integer*4   Nzplast,Nremain
!
!
        write(6,*)  "            "
        write(6,*)  " Enter the number of processors "
        read (5,*)  NumProc
        write(6,*)  " Enter the dimension of the decomp variable"
        read (5,*)  Nz
        write(6,*)  "            "
        write(6,*)  "            "
        write(6,*)  " Job will have ",NumProc," parallel processors "
        write(6,*)  "            "
        write(6,*)  " The decomposition has ",Nz," variables "
        write(6,*)  "            "
        write(6,*)  "            "

!
!      basic processor has  Nz/Numproc variables
!
       Nzp = Nz/NumProc
       write(6,*)  "  Each Process has at least ",Nzp," variables "
!
!      is there a remainder ?
!
       Nremain = Nz - Numproc*Nzp
      
       Nzplast = Nzp + Nremain
!
      if( Nremain .eq. 0 ) then
         write(6,*)  "  The decomp is was perfect "
         write(6,*)  "  Each Process has ",Nzp," variables "
       else
         write(6,*)  "  The decomp is not perfect "
         write(6,*) "            "
         write(6,*) "   ",Nremain," variables were left over "
         write(6,*) "    and have to be assigned, "
         write(6,*) "    we make the highest rank larger  "
         write(6,*) "            "
         write(6,*)  "  Each Process has ",Nzp," variables "
         write(6,*)  "  Except the last one which has ",Nzplast
      endif
!
        write(6,*) "            "
        write(6,*) "            "
        write(6,*) "  Now display the segment addresses "
        write(6,*) "  for the Decomp variable"
        write(6,*) "            "
        write(6,*) "  1 to Nz  by processor"
        write(6,*) "            "
        write(6,*) "   Start - Stop Z index values are : total/proc"
        itotsum = 0
        do ip=1,NumProc
        Nztop = 1 + (ip-1)*Nzp
        Nzbtm = Nztop + Nzp-1
        if( ip .eq. NumProc) then
         Nzbtm = Nz - Nzp*rank
        endif
        itot =  Nzbtm - Nztop + 1
        itotsum = itotsum + itot
        write(6,*) "   ",Nztop,Nzbtm,itot
        enddo
        write(6,*) "            "
        write(6,*) " total sum of decomp variables ",itotsum 
        write(6,*) "            "
         
      end
