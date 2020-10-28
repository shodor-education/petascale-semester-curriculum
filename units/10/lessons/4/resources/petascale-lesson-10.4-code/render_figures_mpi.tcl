# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 4: Visualization 2
# File: render_figures_mpi.tcl
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

proc blockdecompose { framecount } {
  set noderank  [parallel noderank]
  set nodecount [parallel nodecount]
  set start [expr round($noderank     * $framecount / $nodecount)]
  set end [expr round(($noderank+1) * $framecount / $nodecount) - 1]
  return [list $start $end]
}

proc testgather { num trajectoryfile } {
  	set noderank  [parallel noderank]
  	# only print messages on node 0
  	if {$noderank == 0} {
    		puts "Testing parallel gather..."
  	}
 	# Do a parallel gather resulting in a list of all of the node names
	set datalist [parallel allgather $num]
 	# only print messages on node 0
 	if {$noderank == 0} {
    	set filej [open "${trajectoryfile}_${noderank}.dat" w]
    	puts "datalist length: [llength $datalist]"
    	puts "datalist: $datalist"
    	foreach dataline $datalist {
     		foreach line $dataline {
       		puts $filej $line
 		}
  	}
  	close $filej
 	}	
}

proc sumreduction { a b } {
	return [expr $a + $b]
}

proc testreduction {} {
  	set noderank  [parallel noderank]
	 # only print messages on node 0
	 if {$noderank == 0} {
	 	puts "Testing parallel reductions..."
	 }
	 parallel allreduce sumreduction [parallel noderank]
}

proc take_picture {args} {
  	global take_picture
 	 # when called with no parameter, render the image
  	if {$args == {}} {
        set f [format $take_picture(format) $take_picture(frame)]
    	# take 1 out of every modulo images
    	if { [expr $take_picture(frame) % $take_picture(modulo)] == 0 } {
      		render $take_picture(method) $f
	if { $take_picture(exec) != {} } {
                set f [format $take_picture(exec) $f $f $f $f $f $f $f $f $f $f]
  		eval "exec $f"
        }
        }
   	# increase the count by one
    	incr take_picture(frame)
    	return
    	}
  	lassign $args arg1 arg2
	# reset the options to their initial stat
	# (remember to delete the files yourself)
	if {$arg1 == "reset"} {
    		set take_picture(frame)  0
    		set take_picture(format) "./animate.%04d.rgb"
    		set take_picture(method) snapshot
    		set take_picture(modulo) 1
        	set take_picture(exec) {}
    		return
    	}
  	# set one of the parameters
	if [info exists take_picture($arg1)] {
        	if { [llength $args] == 1} {
      			return "$arg1 is $take_picture($arg1)"
        	}
    		set take_picture($arg1) $arg2
    		return
    	}
  # otherwise, there was an error
	error {take_picture: [ | reset | frame | format  | \
                               method  | modulo ]}
}

proc mpianalyze { framecount trajectoryfile } {
	set noderank  [parallel noderank]
  	set nodecount [parallel nodecount]
	set block [blockdecompose $framecount]
  	set start [lindex $block 0]
  	set end   [lindex $block 1]
  	set len   [expr $end - $start + 1]
  	parallel barrier; # wait for all nodes to reach this point
  	puts "Node $noderank, frame range: $start to $end, $len frames total"
  	parallel barrier; # wait for all nodes to reach this point
  	mol addfile $trajectoryfile first $start last $end step 1 waitfor all 
  	parallel barrier; # wait for all nodes to reach this point
  	puts "Node $noderank, loaded [molinfo top get numframes]"
  	parallel barrier; # wait for all nodes to reach this point
	# Display settings 
	source my_ss_colors.tcl
	display projection Orthographic
	display resize 400 800
	display shadows on
	display ambientocclusion on
	axes location Off
	color Display Background white
	mol modselect 0 0 protein
	mol modcolor 0 0 Structure
	mol modstyle 0 0 NewCartoon 0.300000 50.000000 4.100000 0
	mol modmaterial 0 0 AOChalky
	mol smoothrep 0 0 3
	display resetview 
	rotate y by -240
	rotate z by -40
	scale by 1.4
	set dir /work/06295/fabiogon/stampede2/render_mpi_stampede2/figs
	set tachyondir /work/06295/fabiogon/stampede2/vmd-1.9.4a43/lib/tachyon
	# Loop over frames and do your analysis here!
	for {set frame 0} {$frame <= [expr $len - 1]} {incr frame 1} {
		set ts [expr $frame + $start]
		set frameinput ${dir}/figure.${ts}.dat
  		take_picture reset
  		take_picture format ${dir}/figure.${ts}.dat
  		take_picture method Tachyon
  		animate goto $frame
		puts "Frame...$frame"
  		take_picture
		exec ${tachyondir}/tachyon_LINUXAMD64 -aasamples 12 -format TARGA ${dir}/figure.${ts}.dat -o ${dir}/figure.${ts}.dat.tga
	}
  	parallel barrier; # wait for all nodes to reach this point  
}

proc analyzetest { fr trajectoryfile } {
  	parallel barrier; # wait for all nodes to reach this point
  	mol new coordinates_no_water.pdb
	animate delete all
  	parallel barrier; # wait for all nodes to reach this point
  	set num [mpianalyze $fr $trajectoryfile]
  	parallel barrier
  	testgather $fr $trajectoryfile
}

set mytraj production_1microsecond_no_water_ions.dcd
set nframes 5000
testreduction 
analyzetest $nframes $mytraj
quit
