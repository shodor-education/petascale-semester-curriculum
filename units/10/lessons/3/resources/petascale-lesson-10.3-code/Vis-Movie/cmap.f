        program cmap
! Blue Waters Petascale Semester Curriculum v1.0
! Unit 10: Productivity and Visualization
! Lesson 3: Visualization 1
! File: Vis-Movie/cmap.f
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
!       build a gray scale color map
!       use a list of floating point numbers and 
!       build a gray scale color map
!
!       convert the range   +2.6 -> 0.0 -> -2.2 into gray scale pixels
!
!       we transform range  +2.6 -> 0.0 -> -2.6 so 0.0 always has the same gray
!
!       A R G B ===   Alpha(Opacity) RED GREEN BLUE
!
!       In this example we set Alpha = "ff" = 255 meaning image is displayed...
!
!      
!
        write(6,*) " Color and Gray map pixel maker... "
!
        dmaxp =  2.6
        dmaxm = -2.2

        pmax  =  max(dmaxp,abs(dmaxm))

        smax  =  2.0e00 * pmax
        smin  =  0.0e00
!
!
!                        hex 
!                          r g b
!        gray scale  =   ff000000 = black
!        gray scale  =   ffffffff = white
!
         write(6,*) "                 A R G B        "
         write(6,*) "gray scale  =   ff000000 = black"
         write(6,*) "gray scale  =   ffffffff = white"
!
         do looplength = 1,16,15 

         sgray = 0.0
         dgray = 1.0 + (looplength-1)

         smov  = -0.5*smax
          dmov  = smax/256.0
         if( looplength .eq. 16 ) then
          dmov  = smax/16.0
         endif

         write(6,*) "                 "
         write(6,210) 
         write(6,211) 
           do i = 1, 255+looplength, looplength
             ibit = i
             if( ibit .ge. 255 ) ibit = 254
             if( sgray .ge. 255.0 ) sgray = 254.0

             ir   = ibit
             ig   = ibit
             ib   = ibit
             irgb = 256*256*ir + 256*ig + ib

             mone = -256*256*256
             mrgb = mone + irgb
             write(6,200)  sgray,smov,ibit,ir,ig,ib,irgb,mrgb
             sgray = sgray + dgray
             smov  = smov  + dmov 
           enddo
         write(6,*) "                 "
         write(6,*) "                 A R G B        "
         write(6,*) "gray scale  =   ff000000 = black"
         write(6,*) "gray scale  =   ffffffff = white"
         write(6,*) "                 "
         enddo

 200     format(2(f9.3),3x,i4,2x,i4,2x,i4,2x,i4,2x,i8,2x,z8.8)
 201     format(z8.8,3x,z8.8)
 210     format(27x,"                ",10x,"32 Bit Pixel")
 211     format(27x," Red  Green Blue",13x,"A R G B")

         end

     
