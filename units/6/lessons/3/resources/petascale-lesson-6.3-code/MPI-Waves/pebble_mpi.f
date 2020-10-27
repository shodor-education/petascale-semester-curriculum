        program main_pebble_mpi
! Blue Waters Petascale Semester Curriculum v1.0
! Unit 5: MPI
! Lesson 10: Wave Propagation in MPI
! File: MPI-Waves/pebble_mpi.f
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
!       Graphics Magick compatible
!
!
!         Acoustic Wave Propagation
!
!
        include  'mpif.h'
!
        integer*4  rank
        integer*4  size
!
!       mpi error flag needs a name
!
        integer*4  ierror
!
!       systems calls need tags to keep them separate
!
        integer*4  irtag
        integer*4  istag
!
!
!       status is defined and MPI_STATUS_SIZE comes from the mpif.h file referenced above.
!
!
        integer*4  status(MPI_STATUS_SIZE)

!
!
!         Solve the acoustic wave equation in the time domain.
!
!         This is a second order partial differential equation.
!         The domain is two dimensional in space.
!
!         The algorithm is explicit in time and space.  Suitable for
!         single precision float point (32 bit) operations.
!
!         Time starts at zero and waves start moving when
!         the physical domain is excited.  A pebble hits the water surface.
!
!         The medium velocity field is constant, but an array is assigned
!         to this field to allow for experiments with variable velocities.
!
!         The Pebble(t) - is a time field excitation - we make a simple
!         wavelet - something like a spike - and add it to the 
!         center of the wavefield.  It then creates an outgoing wave.
!         The Ricker Wavelet function is used.
!
!         The grid spacing and time interval require control over
!         total time interval of the simualtion.  IF this is not
!         done correctly the method can become unstable.
!
!         Output file is written to allow a animation to be
!         constructed as a post processing step.
!
          parameter (ndx=512,ndz=512)

          parameter (ndtime=1600)

          real*4   Unew   (-1:ndx+2,-1:ndz+2)
          real*4   Ucur   (-1:ndx+2,-1:ndz+2)
          real*4   Uold   (-1:ndx+2,-1:ndz+2)
          real*4   Umovie (-1:ndx+2,-1:ndz+2)

          real*4      frame_float(ndx,ndz)
          integer*4   frame_pixel(ndx,ndz)

          real*4   Velocity(-1:ndx+2,-1:ndz+2)

!     the source term is padded by 500 time steps
          real*4   Pebble_src(ndtime+500)
!     Ricker wavelet source
          integer*4 npoints
          real*4    ricker(10000)
          real*4    frequency
          real*4    dt

          character*25  filename
          character*25  tempfilename
!
!      define image file name character string length 
          nfchar = 25
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
!     construct the decompostion of the ndz variable
!
!
!     remember this works on all processors
!
!     so the last rank machine will have a different number of variables
!
!         real*4   Unew(-1:ndx+2,-1:ndz+2)
!
!        z dimension goes from -1 to ndz+2
!
!        have to pad by 4 to use correct length
!
          nxleft = -1
          nxright = ndx+2

          ndzt = ndz+4
          lndx = ndx+4

          npdz = ndzt/size

          nremain =  ndzt-npdz*size
!
         kztop = -1 + rank*npdz
         kzbtm =  kztop + npdz - 1
!
!    if we are on the highest rank processor then do the following
!    add the pad for the remaining words
!
         if( rank .eq. size-1 ) then
           kzbtm =  kzbtm + nremain
         endif
!

!
!         initialize all wavefield arrays to zero
!

          do j=-1,ndz+2 
!         do j=kztop,kzbtm
           do i=-1,ndx+2
            Unew(i,j) = 0.0e00
            Ucur(i,j) = 0.0e00
            Uold(i,j) = 0.0e00
            Umovie(i,j) = 0.0e00
           enddo
          enddo

          do j=1,ndz
           do i=1,ndx
            frame_float(i,j) = 0.0e00
            frame_pixel(i,j) = 0
           enddo
          enddo
       
!
!         initialize Pebble array to zero
!
           do it=1,ndtime+500
            Pebble_src(it) = 0.0e00
           enddo

!
!         Set the wave field velocity to 1000.0
!
          vel = 1000.0
!
          do j=1,ndz 
!         do j=kztop,kzbtm
           do i=1,ndx
            Velocity(i,j) = vel
           enddo
          enddo
!
!      A movie file is developed during the run.
!
!      Every Nframe is written to a file....
!
          Nframe = 16
          kframe = 0
          
!
!         kframe is the counter of all frames in movie
!
!         iframe is the counter of computations to next frame
!         output.  
!
          iframe = 16
!
!
!         put the temp files into a working temp directory

          tempfilename(1:20) ="temp_fort/rank000000"
          write(tempfilename(15:20),250) rank
 250      format(i6.6)
          write(6,*) " open ",rank,tempfilename(1:20)
          open(12+rank,file=tempfilename(1:20),form="unformatted",
     x     status="new")
!
!
!
!         initialize Pebble time array to a simple zero sum spike
!         over  9 time steps or
!         over 17 time steps.
!
!           Pebble_src(1) =  0.1e00
!           Pebble_src(2) =  0.0e00
!           Pebble_src(3) = -0.6e00
!           Pebble_src(4) =  0.0e00
!           Pebble_src(5) =  1.0e00
!           Pebble_src(6) =  0.0e00
!           Pebble_src(7) = -0.6e00
!           Pebble_src(8) =  0.0e00
!           Pebble_src(9) =  0.1e00
!
!           Pebble_src(1) =  0.1e00
!           Pebble_src(2) = -0.2e00
!           Pebble_src(3) = -0.4e00
!           Pebble_src(4) = -0.5e00
!           Pebble_src(5) = -0.6e00
!           Pebble_src(6) = -0.4e00
!           Pebble_src(7) =  0.2e00
!           Pebble_src(8) =  0.6e00
!           Pebble_src(9) =  1.0e00
!           Pebble_src(10) =  0.6e00
!           Pebble_src(11) =  0.2e00
!           Pebble_src(12) = -0.4e00
!           Pebble_src(13) = -0.6e00
!           Pebble_src(14) = -0.5e00
!           Pebble_src(15) = -0.4e00
!           Pebble_src(16) = -0.2e00
!           Pebble_src(17) =  0.1e00

!
!          initialize Pebble source time array to a simple Ricker wavelet
!          over 400 time steps. Frequency = 30.0 Hertz.
!
!     source
           npoints = 400
          
           frequency = 30.0e00
           dt = 0.0025e00
           call source(npoints,ricker,frequency,dt)
           do i=1,npoints
               Pebble_src(i) = ricker(i)
           enddo
!
!
!         Numerical Stability requires the maximum velocity
!         times the time step to have a bound.  
!         This is know as the Courant Condition.
!
!         Vmax*dt/h < square root (1/2)
!
!         Using this inequality we can compute a time step that
!         allows a stable computation.
!
!         dt    = 0.707 * h / vel
          h     = 3.0
          dt    = 0.45*0.7070 * 3.0 /1000.0
          alpha = vel*vel*dt*dt/h/h
!
!       We have to define the spatial second derivative weights.
!
           w0 =  -2.0e00*205.0e00/  72.00e00
           w1 =            8.0e00/   5.00e00
           w2 =           -1.0e00/   5.00e00
           w3 =            8.0e00/ 315.00e00
           w4 =           -1.0e00/ 560.00e00
!
!          Pebble is placed in the center of the Pond..
! 
           ipeb = ndx/2
           kpeb = ndz/2
 
          if( rank .eq. 0 ) then
           write(6,*) "  Start time loop  "
          endif

!
!      sync before time loop
!
        call MPI_Barrier(MPI_COMM_WORLD,ierr)
!
          do it=1,Ndtime
!
!
!    write the initial quiet water movie frame, because we set
!         iframe = 4
!
          if( iframe .eq. Nframe ) then
!
!    make movie range (less than or equal to 1.0)

!     if( rank .eq. 0 ) write(6,*) rank," normalize movie frame "
!

          ucmax = 0.0e00
          do j=-1,ndz+2 
!         do j=kztop,kzbtm
           do i=-1,ndx+2
            if( abs(Ucur(i,j)) .gt. ucmax) ucmax = abs(Ucur(i,j))
           enddo
          enddo
 
!         write(6,*) rank,ucmax
!
!
!       these max values are local -- need to gather 
!         and broadcast back so all processors are in sync.
!
         call MPI_ALLREDUCE(ucmax,ucmaxg,1,MPI_FLOAT,MPI_MAX,
     x         MPI_COMM_WORLD,ierr) 
!
!         if( rank .eq. 0 ) write(6,*) " all reduce completed "
!         write(6,*) ucmax,ucmaxg,"  max value in movie frame "
!
!         scale - but NOT if ucmax is all ready really small
!
          if( abs(ucmaxg) .gt. 0.000000001 ) then
          ucmaxr = 1.0e00/ucmaxg
          ucmaxr = 1.0e00

          do j=-1,ndz+2 
!         do j=kztop,kzbtm
           do i=-1,ndx+2
            Umovie(i,j) = Ucur(i,j)*ucmaxr
           enddo
          enddo
          endif
!
!          now movie loop is a problem.....
!          need to do all output disk i/o from
!          rank 0...
!
!          we write a temp file and process at end
!          of run -- then move to a real movie file
!
!          Umovie needs to be a big array...
!          so it can hold all processors contribution
!          to the movie frame
!
!       we skip writing the edges
!
!       only write   1 to ndx  and 1 to ndz
!

           kframe = kframe + 1
           kztopw = max(-1,kztop)
           kzbtmw = min(ndz-2,kzbtm)
           write(12+rank) rank,ndx,kztopw,kzbtmw
!          write(6,*) rank,ndx,kztopw,kzbtmw
        do ix=1,ndx
         write(12+rank) (Umovie(ix,kz),kz=kztopw,kzbtmw)
        enddo
!       write(32+rank,*) rank,(ndx*(kzbtmw-kztopw+1)+3+2+2)*4

        itemp = (ndx*(kzbtmw-kztopw+1)+3)*4
!       write(32+rank,*) rank,itemp
!   we did a write so reset iframe counter
            iframe = 0
            else
            iframe = iframe + 1

          endif
!

!       we can use IF statement in the loops to add the source term
!       we can create an entire array for the source term
!       or we can add the source term to Ucur and then subtract 
!         at the end of the time loop - remove it saving an IF inside
!         the loop....

           kztopt = max(5,kztop)
           kzbtmt = min(kzbtm,ndz-4)

!
!        Add the Pebble excitation from the current wave field.
!        do this in the correct memory block
!
         if( kpeb .ge. kztopt ) then 
          if( kpeb .le. kzbtmt ) then 
           Ucur(ipeb,kpeb) = Ucur(ipeb,kpeb) + Pebble_src(it)
           if( it .gt. 25 ) then
            Ucur(ipeb-60,kpeb) = Ucur(ipeb-60,kpeb) + Pebble_src(it-25)
           endif
          endif
         endif

!
!
!      Before we do a time step make sure all halo block zones
!      are copied
!
!        we do 5 rows of x from -1 to ndx+2
!
!     skip the last block as it has no data to send
!     inside the model.
!
!
!     first copy down in Z
!
         lucur = 1*lndx
         lucur = 5*lndx
!        write(6,*) "                        "
!        write(6,*) "Message pass to,",rank+1," from ",rank
!        write(6,*) "Moving ",lucur," words as floats " 

        call MPI_Barrier(MPI_COMM_WORLD,ierr)
      if( rank .eq. 0) then
!        write(6,*)  Ucur(100,  kzbtm-4)
!        write(6,*)  Uold(100,  kzbtm-4)
      endif

!
!    move top overlay zone to left
!    move top overlay zone to right
!
!
!               rank         rank
!                 0   <--     1
!                 0   -->     1
!
         if( rank .gt. 0 ) then
          irkto = rank-1
          istag = 1000+rank
          call MPI_Send(Ucur(nxleft, kztop),lucur,
     x             MPI_Float,irkto,istag,
     x             MPI_COMM_WORLD,ierr) 
         endif

         if( rank .le. size-2 ) then
          irfrom = rank +1
          irtag = 1000+rank+1
          call MPI_Recv(Ucur(nxleft, kzbtm+1),lucur,
     x             MPI_Float,irfrom,irtag,
     x             MPI_COMM_WORLD,status,ierr) 
         endif

         if( rank .gt. 0 ) then
          irkto = rank-1
          istag = 3000+rank
          call MPI_Send(Uold(nxleft, kztop),lucur,
     x             MPI_Float,irkto,istag,
     x             MPI_COMM_WORLD,ierr) 
         endif

         if( rank .le. size-2 ) then
          irfrom = rank +1
          irtag = 3000+rank+1
          call MPI_Recv(Uold(nxleft, kzbtm+1),lucur,
     x             MPI_Float,irfrom,irtag,
     x             MPI_COMM_WORLD,status,ierr) 
         endif

!
!    move bottom overlay zone to left
!    move bottom overlay zone to right
!
!
!               rank         rank
!                 0   -->     1
!                 0   <--     1
!
         if( rank .le. size-2 ) then
          irkto = rank+1
          istag = 2000+rank+1
          call MPI_Send(Ucur(nxleft, kzbtm-4),lucur,
     x             MPI_Float,irkto,istag,
     x             MPI_COMM_WORLD,ierr) 
         endif

         if( rank .gt. 0 ) then
          irfrom = rank-1
          irtag = 2000+rank
          call MPI_Recv(Ucur(nxleft, kztop-5),lucur,
     x             MPI_Float,irfrom,irtag,
     x             MPI_COMM_WORLD,status,ierr) 
         endif

         if( rank .le. size-2 ) then
          irkto = rank+1
          istag = 4000+rank+1
          call MPI_Send(Uold(nxleft, kzbtm-4),lucur,
     x             MPI_Float,irkto,istag,
     x             MPI_COMM_WORLD,ierr) 
         endif

         if( rank .gt. 0 ) then
          irfrom = rank-1
          irtag = 4000+rank
          call MPI_Recv(Uold(nxleft, kztop-5),lucur,
     x             MPI_Float,irfrom,irtag,
     x             MPI_COMM_WORLD,status,ierr) 
         endif

!
!
!
        call MPI_Barrier(MPI_COMM_WORLD,ierr)
!
!     next copy up in Z
!
!         write(6,*) it," time step "
!
  
!          do kz = 5,ndz-4
           do kz=kztopt,kzbtmt
            do ix = 5,ndx-4
             Unew(ix,kz) = 2.0*Ucur(ix,kz) -Uold(ix,kz) + alpha*
     x (w0*Ucur(ix,kz)+
     x  w1*(Ucur(ix-1,kz)+Ucur(ix+1,kz)+Ucur(ix,kz-1)+Ucur(ix,kz+1))+ 
     x  w2*(Ucur(ix-2,kz)+Ucur(ix+2,kz)+Ucur(ix,kz-2)+Ucur(ix,kz+2))+ 
     x  w3*(Ucur(ix-3,kz)+Ucur(ix+3,kz)+Ucur(ix,kz-3)+Ucur(ix,kz+3))+ 
     x  w4*(Ucur(ix-4,kz)+Ucur(ix+4,kz)+Ucur(ix,kz-4)+Ucur(ix,kz+4)) )
            enddo
           enddo

!
!        remove the Pebble excitation from the current wave field.
!        do this in the correct memory block
!       
         if( kpeb .ge. kztopt ) then 
          if( kpeb .le. kzbtmt ) then 
           Ucur(ipeb,kpeb) = Ucur(ipeb,kpeb) - Pebble_src(it)
           if( it .gt. 25 ) then
            Ucur(ipeb-60,kpeb) = Ucur(ipeb-60,kpeb) - Pebble_src(it-25)
           endif
          endif
         endif
!
!      We have advanced a time step..
!
!      now we have to move Current time to Old time
!      now we have to move New time to Current time
!
           do ix = 1,ndx
!           do kz = 1,ndz
            do kz=kztop,kzbtm
             Uold(ix,kz) = Ucur(ix,kz)
            enddo
           enddo

           do ix = 1,ndx
!           do kz = 1,ndz
            do kz=kztop,kzbtm
             Ucur(ix,kz) = Unew(ix,kz)
            enddo
           enddo
!
!       end of time loop
!
          enddo
!
!     at end of time loop we write one more record to 
!     temp file to mark the end of data
!     we mark end of file with a big negative number -999
!
      ienddata = -999
!
!     write(12+rank) irank,ienddata,ienddata,ienddata
!     write(6,*) "EOF Marker",rank,irank,ienddata,ienddata,ienddata

        if( rank .eq. 0 ) then
	  write(6,*) "             "
          write(6,*) " Movie has ",kframe," Frames "
          write(6,*) " Each frame has ",ndz," rows "
          write(6,*) " Each row has ",ndx," floating point numbers "
      write(6,*)" Each unformatted floating point number has 32 bits"
          write(6,*) "             "
        endif


!
!     pause and sync all processes to a halt
!     then finish
!
        call MPI_Barrier(MPI_COMM_WORLD,ierr)
!
!       beginning of rank = 0 file consolidation
!
         if( rank .eq. 0) then
!
!         Movie file record length is in bytes
!         Single file record is a complete wavefield image
!         without the pads.
!
!         open file for movie on rank = 0
!
!
          Nrecl = 4*ndz
          open(10,file="data/pebble.mvlp",form="unformatted",
     *         status="unknown",
     *         access="direct",recl=Nrecl)
!
!
!         file output record for movie loop is initialized
!
          irec = 0

       mvframe = 0
!
!      close temp files 
!
           do irnk=1,size
             close (12+irnk-1)
           enddo
!
!
!         now reopen the temp files from a working temp directory

         do irnk=1,size

          tempfilename(1:20) ="temp_fort/rank000000"
          write(tempfilename(15:20),250) irnk-1
          write(6,*) " open ",irnk-1,tempfilename(1:20)
          open(12+irnk-1,file=tempfilename(1:20),form="unformatted",
     x     status="old")

         enddo
!
!

  100     continue
!

           do irnk=1,size

!            ifl = 12+irnk-1
!            write(6,*) " before read of temp file ",ifl

             read(12+irnk-1,end=99) lirank,lndxt,lkztopf,lkzbtmf

!            write(6,*) rank,ifl,lirank,lndxt,lkztopf,lkzbtmf
!
!      see if file is marked with an end of data flag...
             if( lndxt .eq. -999 ) go to 99

             do ix=1,lndxt
              read(12+irnk-1) (Umovie(ix,kz),kz=lkztopf,lkzbtmf)
             enddo

           enddo
!
!           now we have read a time step movie frame
!           write it to a SU random access file
!
!
             mvframe = mvframe + 1

             do ix=1,ndx
              irec=irec+1
              write(10,rec=irec) (Umovie(ix,kz),kz=1,ndz)
             enddo
!
!            go back and get the next movie frame
!
             write(6,240) mvframe,kframe
             go to 100
!
!    end of file read on one of the temp files - so quit
!
  99        continue
!
!     now close all temp files
!
           do irnk=1,size
             close(12+irnk-1)
           enddo

        write(6,240) mvframe,kframe
 240    format(" movie frame counts ",2i6)
!
!
!

!
!       end of rank = 0 file consolidation
!

!     NOW make the pixel movie frames
!
!     only need one copy of these files -- let rank 0 do the write
!
          Nrecl = 4*ndz
          open(10,file="data/pebble.mvlp",form="unformatted",
     *         status="unknown",
     *         access="direct",recl=Nrecl)

!    
!     Build Graphics Magick header file and file name file.
!
      open(9,file="data/pebble.hdr",form="unformatted",
     x       recl=16,
     x       access="direct")
      write(9,rec=1) kframe,ndx,ndz,nfchar
      close(9)

!   character data needs a formatted file
      open(9,file="data/pebble.filenames",form="formatted",
     x       recl=25,
     x       access="direct")
      filename(1:25) = "data/pebble_mvlp00000.dat"
!    filename file
      irecln = 1
!    movie loop file 
      irecmf = 1
      do ifn=1,kframe
       write(filename(17:21),200) ifn-1
 200   format(i5.5)
       write(9,222,rec=irecln) (filename(k:k),k=1,25)
 222   format(25a1)
       irecln = irecln + 1
!
!    now read movie frame and write the .dat file
! 
       do ix=1,ndx
        read(10,rec=irecmf) (frame_float(ix,kz),kz=1,ndz)
        irecmf = irecmf + 1
       enddo
       Nrecl = 4*ndz
       open(8,file=(filename(1:25)),form="unformatted",
     x       recl=Nrecl,
     x       access="direct")
       irecmfn = 0
       call  float_pixel(frame_float,frame_pixel,ndx,ndz)
       do ix=1,ndx
        irecmfn = irecmfn + 1
        write(8,rec=irecmfn) (frame_pixel(ix,kz),kz=1,ndz)
       enddo
       close(8)

      enddo
      close(9)

        endif
!
          close(10)

          call MPI_Barrier(MPI_COMM_WORLD,ierror)

          write(6,*) rank," Program has completed "

!
!     real work is done so shut down....
!
!
!     close out the parallel world is a good idea.
!     otherwise a hanging process can clutter up the operating system.
!

        call MPI_FINALIZE(ierror)

        end
        subroutine source(npoints,ricker,frequency,dt)
!
!         R. Phillip Bording
!         Blue Waters Project
!         July 18, 2020
!       
!
!       create a Ricker wavelet with a center "frequency".
!
!       output is a source vector - array(npoints)
!
!      Hosken, J. W., 1980
!      Ricker wavelets in their various guises. First Break, v. 6(1), p. 24-33.
!

        integer*4  npoints
        real*4     ricker(*)
        real*4     cpift
        real*4     dt
        real*4     frequency
        real*4     time

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
        enddo

        return
        end

        subroutine  float_pixel(frame_float,frame_pixel,ndx,ndz)
!
!      R. Phillip Bording
!      July 29, 2020
!
!      Convert float to pixel a simple rectangular movie frame
!
!
        real*4    frame_float(ndx,ndz)
        integer*4 frame_pixel(ndx,ndz)

        integer*4 frame_int  (ndx,ndz)

!
!      now find the maximum positive value in the float array
!      now find the maximum negative value in the float array
!
       dmaxp = 0.0e00
       dminn = 0.0e00

       do kz=1,ndz
        do ix=1,ndx
      if(frame_float(ix,kz) .ge. dmaxp) dmaxp = frame_float(ix,kz)
      if(frame_float(ix,kz) .le. dminn) dminn = frame_float(ix,kz)
        enddo
       enddo
!
!  We can re-scale the frame if it is not close to 1.0 for a max value
!
       rescale = 1.0e00
       amax = max(dmaxp,abs(dminn))
       if( amax .gt. 0.0000001) then
       rescale = 1.0/amax
!
       do kz=1,ndz
        do ix=1,ndx
          frame_float(ix,kz) = rescale * frame_float(ix,kz)
        enddo
       enddo

       endif
     
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
!                   mfm = (frame_float(ix,kz)+1.0e00)*127.5e00 
!        write(6,*) mfm,(frame_float(ix,kz)+1.0e00)*127.5e00 

      frame_int(ix,kz) = int((frame_float(ix,kz)+1.0e00)*127.5e00) 
       
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
         irgb = frame_int(ix,kz)
         frame_pixel (ix,kz) =  none + 256*256*irgb  + 256*irgb + irgb 
        enddo
       enddo
                 
       return
       end 
