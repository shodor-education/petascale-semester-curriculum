          program main_pebble
! Blue Waters Petascale Semester Curriculum v1.0
! Unit 10: Productivity and Visualization
! Lesson 3: Visualization 1
! File: Vis-Movie/pebble.f
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
!         GraphicsMagick compatible  
!
!         Movie making PDE solver
!
!         Acoustic Wave Propagation
!
!         Solve the acoustic wave equation in the time domain.
!
!         This is a second order partial differential equation.
!         The domain is two dimensional in space.
!
!         The alogrithm is explicit in time and space.  Suitable for
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
!
!         The grid spacing and time interval require control over
!         total time interval of the simualtion.  IF this is not
!         done correctly the method can become unstable.
!
!
!
!         Output files are written to allow a animation to be
!         constructed as a post processing step.
!
!    
!
!
!
          parameter (ndx=512,ndz=512)

          parameter (ndtime=3200)


          real*4   Unew(-1:ndx+2,-1:ndz+2)
          real*4   Ucur(-1:ndx+2,-1:ndz+2)
          real*4   Uold(-1:ndx+2,-1:ndz+2)
          real*4   Umovie(-1:ndx+2,-1:ndz+2)

          real*4      frame_float(ndx,ndz)
          integer*4   frame_pixel(ndx,ndz)

          real*4   Velocity(ndx,ndz)

          real*4   Pebble(ndtime)
!    source
        integer*4  npoints
        real*4     ricker(10000)
        real*4     frequency
        real*4     dt

        character*25  filename

        integer*4  kframe
!
!         initialize all wavefield arrays to zero
!

          do j=-1,ndz+2 
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
           do it=1,ndtime
            Pebble(it) = 0.0e00
           enddo

           kframe = 0
!
!         Set the wave field velocity to 1000.0
!
          vel = 1000.0
!
          do j=1,ndz 
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
          
!
!         Movie file record length is in bytes
!         Single file record is a complete wavefield image
!         without the pads.
!
          Nrecl = 4*ndz
          open(10,file="data/pebble.mvlp",form="unformatted",
     *         status="unknown",
     *         access="direct",recl=Nrecl)
!
!         iframe is the counter of computatoins to next frame
!         output.  
!
          iframe = 16
!
!         file output record for movie loop is initialized
!
          irec = 0
!
!         initialize Pebble time array to a simple zero sum spike
!         over 9 time steps.
!
            Pebble(1) =  0.1e00
            Pebble(2) =  0.0e00
            Pebble(3) = -0.6e00
            Pebble(4) =  0.0e00
            Pebble(5) =  1.0e00
            Pebble(6) =  0.0e00
            Pebble(7) = -0.6e00
            Pebble(8) =  0.0e00
            Pebble(9) =  0.1e00
!
!         initialize Pebble time array to a simple Ricker wavelet
!         over 400 time steps.
!
!    source

        npoints = 400
         
        frequency = 30.0e00
        dt = 0.0025e00
        call source(npoints,ricker,frequency,dt)
        do i=1,npoints
            Pebble(i) =  ricker(i)
        enddo
        
!
!         Numerical Satbility requires the maximum velocity
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
 
!
          do it=1,Ndtime
!
!
!    write the initial quite water movie frame, because we set
!    iframe = 16
!
          if( iframe .eq. Nframe ) then
!
!    make movie range (less than or equal to 1.0)
!

          ucmax = 0.0e00
          do j=-1,ndz+2 
           do i=-1,ndx+2
            if( abs(Ucur(i,j)) .gt. ucmax) ucmax = abs(Ucur(i,j))
           enddo
          enddo
          write(6,*) ucmax,"  max value in movie frame "
!
!         scale - but NOT if ucmax is all ready really small
!
          if( abs(ucmax) .gt. 0.000000001 ) then
          ucmaxr = 1.0e00/ucmax
          do j=-1,ndz+2 
           do i=-1,ndx+2
            Umovie(i,j) = Ucur(i,j)*ucmaxr
           enddo
          enddo
          endif

           kframe = kframe + 1

           do ix=1,ndx
            irec = irec + 1
            write(10,rec=irec) (Umovie(ix,kz),kz=1,ndz)
           enddo
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
!
!        Add the Pebble excitation from the current wave field.
!
           Ucur(ipeb,kpeb) = Ucur(ipeb,kpeb) + Pebble(it)
!
  
           do ix = 5,ndx-4
            do kz = 5,ndz-4
             Unew(ix,kz) = 2.0*Ucur(ix,kz) -Uold(ix,kz) + alpha*
     x (w0* Ucur(ix,kz)+
     x  w1*(Ucur(ix-1,kz)+Ucur(ix+1,kz)+Ucur(ix,kz-1)+Ucur(ix,kz+1))+ 
     x  w2*(Ucur(ix-2,kz)+Ucur(ix+2,kz)+Ucur(ix,kz-2)+Ucur(ix,kz+2))+ 
     x  w3*(Ucur(ix-3,kz)+Ucur(ix+3,kz)+Ucur(ix,kz-3)+Ucur(ix,kz+3))+ 
     x  w4*(Ucur(ix-4,kz)+Ucur(ix+4,kz)+Ucur(ix,kz-4)+Ucur(ix,kz+4)) )
            enddo
           enddo

!
!        remove the Pebble excitation from the current wave field.
!
           Ucur(ipeb,kpeb) = Ucur(ipeb,kpeb) - Pebble(it)
!
!      We have advanced a time step..
!
!      now we have to move Current time to Old time
!      now we have to move New time to Current time
!
           do ix = 1,ndx
            do kz = 1,ndz
             Uold(ix,kz) = Ucur(ix,kz)
            enddo
           enddo

           do ix = 1,ndx
            do kz = 1,ndz
             Ucur(ix,kz) = Unew(ix,kz)
            enddo
           enddo
!
!       end of time loop
!
          enddo

          write(6,*) "             "
          write(6,*) " Movie has ",kframe," Frames "
          write(6,*) "             "


          close(10)
          Nrecl = 4*ndz
          open(10,file="data/pebble.mvlp",form="unformatted",
     *         status="unknown",
     *         access="direct",recl=Nrecl)

       nfchar = 25
!
!      Build GraphicsMagick header file and file name file.
!
          open(9,file="data/pebble.hdr",form="unformatted",
     x           recl=16,
     x           access="direct")
          write(9,rec=1)  kframe,ndx,ndz,nfchar
          close(9)

!       character data needs a formatted file
          open(9,file="data/pebble.filenames",form="formatted",
     x           recl=25,
     x           access="direct")
          filename(1:25) = "data/pebble_mvlp00000.dat"
!    file name file
          irecln= 1
!    movie loop file
          irecmf = 1
          do ifn=1,kframe
           write(filename(17:21),200) ifn-1
 200       format(i5.5)
           write(9,222,rec=irecln)  (filename(k:k),k=1,25)
 222       format(25a1)
           irecln=irecln + 1
!
!    now read movie frame and write the .dat file
!
           do ix=1,ndx
            read(10,rec=irecmf) (frame_float(ix,kz),kz=1,ndz)
            irecmf = irecmf + 1
           enddo
           Nrecl = 4*ndz
           open(8,file=(filename(1:25)),form="unformatted",
     x           recl=Nrecl,
     x           access="direct")
           irecmfn = 0
           call  float_pixel(frame_float,frame_pixel,ndx,ndz)
           do ix=1,ndx
            irecmfn = irecmfn + 1
            write(8,rec=irecmfn) (frame_pixel(ix,kz),kz=1,ndz)
           enddo
           close(8)

          enddo
          close(9)

          write(6,*) " Program has completed "

          end
        subroutine source(npoints,ricker,frequency,dt)
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
        write(6,*) it,time,ricker(it)
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
        

