# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 4: Visualization 2
# File: addpalette.tcl
# Developed by Juan R. Perilla for the Shodor Education Foundation, Inc.
#
# Copyright (c) 2020 The Shodor Education Foundation, Inc.
#
# Browse and search the full curriculum at
# <http://shodor.org/petascale/materials/semester-curriculum>.
#
# We welcome your improvements! You can submit your proposed changes to this
# material and the rest of the curriculum in our GitHub repository at
# <https://github.com/shodor-education/petascale-semester-curriculum>.
#
# We want to hear from you! Please let us know your experiences using this
# material by sending email to petascale@shodor.org
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Color Palette Procedures

proc cmyk2rgb { c m y k } {
    set R [expr (1-$c)*(1-$k)]
    set G [expr (1-$m)*(1-$k)]    
    set B [expr (1-$y)*(1-$k)]
    return "$R $G $B"
}

proc lab2xyz { L a b} {
    set Y [expr ($L + 16) /116]
    set X [expr $a /500 + $Y]
    set Z [expr $Y - $b /200]
    if [expr $Y**3 > 0.008856] {
	set Y [expr $Y**3]
    } else {	      
	set Y [expr ($Y - 16 /116)/7.787]
    } 
    if [expr $X**3 > 0.008856] {
	set X [expr $X**3]
    } else {	      
	set X [expr ($X - 16 /116)/7.787]
    } 
    if [expr $Z**3 > 0.008856] {
	set Z [expr $Z**3]
    } else {	      
	set Z [expr ($Z - 16 /116)/7.787]
    } 
    # Observer= 2°, Illuminant= D65
    set refX 95.047
    set refY 100.000
    set refZ 108.883

    set X [expr $refX * $X]
    set Y [expr $refY * $Y]
    set Z [expr $refZ * $Z]

    return "$X $Y $Z"
}

proc xyz2rgb { X Y Z } {
  #  (Observer = 2°, Illuminant = D65)
    set X [expr $X / 100]
    set Y [expr $Y / 100]   
    set Z [expr $Z / 100]

    set R [expr $X * 3.2406 + $Y * - 1.5372 + $Z * -0.4986]
    set G [expr $X * -0.9689 + $Y * 1.8758 + $Z * 0.0415]
    set B [expr $X * 0.0557  + $Y * - 0.2040 + $Z * 1.0570]
    
    if [expr $R > 0.0031308] { 
	set R [expr 1.055 * ($R ** (1/2.4)) - 0.055] 
    } else {
	set R [expr 12.92 * $R]
    }

    if [expr $G > 0.0031308] { 
	set G [expr 1.055 * ($G ** (1/2.4)) - 0.055] 
    } else {
	set G [expr 12.92 * $G]
    }

    if [expr $B > 0.0031308] { 
	set B [expr 1.055 * ($B ** (1/2.4)) - 0.055] 
    } else {
	set B [expr 12.92 * $B]
    }

    return "$R $G $B"
}

proc lab2rgb { L a b } {
    lassign [lab2xyz $L $a $b] x y z
    return [xyz2rgb $x $y $z]
}

proc changeColor { i L a  b } {
    lassign [lab2rgb $L $a $b] r g b
    color change rgb $i $r $g $b  
}


proc changeCapsidColors { col1 col2 col3 col4 } {
    set NTD 7
    set CTD 2
    set PEN 3
    set CYP 9
    lassign $col1 L a b
    changeColor $NTD $L $a $b
    lassign $col2 L a b
    changeColor $CTD $L $a $b
    lassign $col3 L a b
    changeColor $PEN $L $a $b
}

proc changeCapsidColorsRGB { col1 col2 col3 col4 } {
    set NTD 7
    set CTD 2
    set PEN 3
    set CYP 9
    if {[scan $col1 "#%2x%2x%2x" r g b] != 3} {
	lassign $col1 r g b
    }
    set r [expr $r / 255.]
    set g [expr $g / 255.]
    set b [expr $b / 255.]
    color change rgb $NTD $r $g $b
    if {[scan $col2 "#%2x%2x%2x" r g b] != 3}   {
	lassign $col2 r g b
    }

    set r [expr $r / 255.]
    set g [expr $g / 255.]
    set b [expr $b / 255.]
    color change rgb $CTD $r $g $b
    if {[scan $col3 "#%2x%2x%2x" r g b] != 3}   {
	lassign $col3 r g b
    }
    set r [expr $r / 255.]
    set g [expr $g / 255.]
    set b [expr $b / 255.]
    color change rgb $PEN $r $g $b

    if {[scan $col4 "#%2x%2x%2x" r g b] != 3}   {
	lassign $col4 r g b
    }
    set r [expr $r / 255.]
    set g [expr $g / 255.]
    set b [expr $b / 255.]
    color change rgb $CYP $r $g $b

}




proc changeColorRGB { vmdcolor col1  } {
    if {[scan $col1 "#%2x%2x%2x" r g b] != 3} {
	lassign $col1 r g b
    }
    set r [expr $r / 255.]
    set g [expr $g / 255.]
    set b [expr $b / 255.]
    color change rgb $vmdcolor $r $g $b

}

if {0} {

# Beach colors  - 8.0
set col1 "42.45 -4.85 -23.67"
set col2 "81.3 -3.24 -5.8"
set col3 "79.7 13.41 67.53"

changeCapsidColors $col1 $col2 $col3
take_picture

# Floral colors - 5.0

set col1 "80.71 -19.12 34.7"
set col2 "40.59 -24.11 41.63"
set col3 "80.19 -5.64 -19.53"

changeCapsidColors $col1 $col2 $col3
take_picture

# Morning dew. The hood of my car. A poor lady bug. 9.0
# Colors used by Russian team that won S&E image award (DO NOT USE)

set col1 "56.78 -0.64 0.78"
set col2 "38.02 0.32 0.16"
set col3 "49.42 47.82 49.29"

changeCapsidColors $col1 $col2 $col3
take_picture
# Silly prtrait of a woman enjoying pi. 7.0

set col1 "81.18 6.19 12.2"
set col2 "50.08 27.59 41.3"
set col3 "40.05 56.53 40.16"
changeCapsidColors $col1 $col2 $col3
take_picture
# Graffiti Artist sprays mural on a wall dedicated to graffiti Art.
# 6.0

set col1 "80.99 14.8 16.84"
set col2 "34.88 -0.09 0.62"
set col3 "21.58 23.27 -38.89"
changeCapsidColors $col1 $col2 $col3
take_picture
# The giant 'golden dome' of the Archer Daniels Midland Co-Generation Plant in downtown Clinton, IA looms over the town as an eastbound Union Pacific coal train leaves town headed for Chicago. The dome 
# 5.0

set col1 "70.9 22.15 43.51"
set col2 "39.95 13.78 21.73"
set col3 "29.87 -2.82 -17.34"
set col4 "21.58 23.27 -38.89"
changeCapsidColors $col1 $col2 $col3 $col4
take_picture
# a sliced pomegranate in Israel.
# 7.5

set col1 "61.12 25.86 7.96"
set col2 "19.73 19.36 1.83"
set col3 "29.9 36.66 15.69"
changeCapsidColors $col1 $col2 $col3
take_picture
# outh America, Argentina, La Plata, Los Olmos. Freedom Behind Bars, Prison and Divinity - Sleeping and living quarters in Unit 25 of Los Olmos Prison. Los Olmos Prison is one of the principal security

set col1 "70.36 6.95 27.37"
set col2 "50.89 7.33 8.75"
set col3 "19.73 19.36 1.83"
changeCapsidColors $col1 $col2 $col3
take_picture


####
#
# RGBs
#
####

# Art Attack - Paul Frank
# 9.0

set col1 "91 197 191"
set col2 "249 236 209"
set col3 "237 29 36"
changeCapsidColorsRGB $col1 $col2 $col3
take_picture
# Guerra creativa

set col1 "252 124 61"
set col2 "116 0 37"
set col3 "206 0 39"
changeCapsidColorsRGB $col1 $col2 $col3
take_picture
# Reading Ernst
# too yellow 

set col1 "255 203 37"
set col2 "255 99 37"
set col3 "235 27 0"
changeCapsidColorsRGB $col1 $col2 $col3
take_picture

#Ernst fuchs

set col1 "163 192 162"
set col2 "80 139 133"
set col3 "253 200 88"
changeCapsidColorsRGB $col1 $col2 $col3
take_picture
# Pierrot

set col1 "140 177 170"
set col2 "79 47 52"
set col3 "232 105 36"
changeCapsidColorsRGB $col1 $col2 $col3 $col3
take_picture
# Tarantula

set col1 "\#b9af40"
set col2 "\#a29b71"
set col3 "\#c32c7b"
changeCapsidColorsRGB $col1 $col2 $col3
take_picture
# CSD #1

set col1 "\#f2c597"
set col2 "\#315a9a"
set col3 "\#92b979"

changeCapsidColorsRGB $col1 $col2 $col3
take_picture

# NVidia

set col1 "\#8bcb12" 
set col2 "\#6f6f6f"
set col3 "\#a8ea31"

changeCapsidColorsRGB $col1 $col2 $col3
take_picture


# Cyclophilin


set col1 "\#ff8e46"
set col2 "\#CC9e8f"
set col3 "\#00917f"
set col4 "\#ffcbd6"
changeCapsidColorsRGB $col1 $col2 $col3 $col4


set NTD "\#fbb47e"
set CTD "\#1b3f33"
set PEN "\#0c211a"
set CYP "\#f66522"
changeCapsidColorsRGB $NTD $CTD $PEN $CYP

set NTD "\#dbcd8e"
set CTD "\#2b1210"
set PEN "\#ff1800"
set CYP "\#7cbc78"
changeCapsidColorsRGB $NTD $CTD $PEN $CYP

set NTD "\#d2e2e2"
set CTD "\#335e64"
set PEN "\#009fab"
set CYP "\#c72300"
changeCapsidColorsRGB $NTD $CTD $PEN $CYP

### hbv COLORS

set NTD "\#90ddee"
set CTD "\#286d7c"
set PEN "\#ec5656"
set CYP "\#f8e9d3"
changeCapsidColorsRGB $NTD $CTD $PEN $CYP

}
