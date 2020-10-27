              program hello_world
! Blue Waters Petascale Semester Curriculum v1.0
! Unit 5: MPI
! Lesson 10: Wave Propagation in MPI
! File: MPI-Waves/hello_world.f
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
!       Setup a parallel program just to see if MPI works.
!
!
        include  'mpif.h'
        
!
!      internal variables needed to define the
!
!       size -- total number of processors
!       rank -- 0 to size-1 -- my location in the processor scheme
!
!       size = 4   means we have 4 processors computing
!
!       rank is then the set of {0,1,2,3) processors doing the computing
!
!
        integer*4  rank
        integer*4  size
!
!       error flag needs a name
!
        integer*4  ierror
!
!       systems call need tags to keep them separate
!
        integer*4  tag
!
!
!       status is defined and MPI_STATUS_SIZE comes from the mpif.h file referenced above.
!
!
        integer*4  status(MPI_STATUS_SIZE)

 
!
!      wake up MPI
!
        call MPI_INIT(ierror)
!
!     find out how many will do the work??
!
        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
!
!     find out who I am in the grand scheme of things?
!
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)
!
!     now do real work
!
!
!     so every process does a print...  
!

        write(6,*) " Hello from ",rank," of ",size," processes "

        do i=1,3
         write(12+rank,*) " Hello from ",rank," of ",size," processes "
        enddo
!
!
!       now we let process 0 = rank = 0 tell us what is going on....
!
!       This way we keep form multiling the i/o out from the program..
!
!
      if( rank .eq. 0 ) then
        write(6,*) " Hello from rank = zero of ",size," processes "
      endif


!
!     real work is done so shut down....
!
!
!     close out the parallel world is a good idea.
!     otherwise a hanging process can clutter up the operating system.
!

        call MPI_FINALIZE(ierror)

        end
