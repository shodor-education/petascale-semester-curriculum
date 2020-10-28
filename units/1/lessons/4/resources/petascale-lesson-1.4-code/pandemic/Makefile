# Blue Waters Petascale Semester Curriculum v1.0
# Unit 1: OpenMP
# Lesson 4: Scientific Examples on a Single Core
# File: Makefile
# Developed by Aaron Weeden for the Shodor Education Foundation, Inc.
# Modified by Yu Zhao, Macalester College (Modularized and restructured the
# original code)
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

# Code prefix
PROGRAM_PREFIX=pandemic

# Compilers and flags
CC=gcc
#XLIB_LOC=/opt/X11/lib    #Mac OS X XQuartz installed here
#XLIB_LOC=/usr/X11R6/lib   #some unix systems may have this
#XLIB_INC=/opt/X11/include

ifdef ICC
	CC=icc
endif

#CFLAGS+=-DTEXT_DISPLAY # Uncomment to show text display

CFLAGS+=-DX_DISPLAY -I $(XLIB_INC) -L$(XLIB_LOC) -lX11 # Uncomment to show X display

CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SRCS=$(PROGRAM_PREFIX).c

# Make targets
all: $(PROGRAM_PREFIX)-serial

clean: 
	rm -f $(PROGRAM_PREFIX)_serial

run:
	./$(PROGRAM_PREFIX)_serial

# Make rules
$(PROGRAM_PREFIX)-serial: $(SRCS)
	$(CC) -o $(PROGRAM_PREFIX)_serial $(SRCS) $(CFLAGS) 

$(SRCS): Core.h Defaults.h Display.h Finalize.h Infection.h Initialize.h
