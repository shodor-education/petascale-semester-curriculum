# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 4: Visualization 2
# File: my_ss_colors.tcl
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

source addpalette.tcl

proc reset_sscolors {} {
    color Structure "Alpha Helix" purple
    color Structure 3_10_Helix blue
    color Structure Pi_Helix red
    color Structure Extended_Beta yellow
    color Structure Bridge_Beta tan
    color Structure Turn cyan
    color Structure Coil white
}

proc new_sscolors { colors } {
    lassign $colors col1 col2 col3 col4 col5 col6 col7
    set colorlist [list "\"Alpha Helix\" 22 $col1" "\"3_10_Helix\" 23 $col2" "\"Pi_Helix\" 24 $col3" "\"Extended_Beta\" 25 $col4" "\"Bridge_Beta\" 26 $col5" "\"Turn\" 27 $col6" "\"Coil\" 28 $col7"]

  foreach color $colorlist {
    lassign $color structure colNum colRGB
    puts "Changing color $colNum $structure $colRGB"
    color Structure $structure $colNum
    changeColorRGB $colNum $colRGB
  }
}


new_sscolors {"\#5FC2D9" "\#5FFFB9" "\#9BC1BC" "\#F28E85" "\#9BC1BC" "\#FFF0D9" "\#FFF0D9" }
