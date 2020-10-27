# Blue Waters Petascale Semester Curriculum v1.0
# Unit 10: Productivity and Visualization
# Lesson 3: Visualization 1
# File: mimage/Makefile
# Developed by R. Phillip Bording for the Shodor Education Foundation, Inc.
# Included in the following lessons:
# - Unit 5 (MPI) Lesson 10: Wave Propagation in MPI
# - Unit 6 (Hybrid MPI + OpenMP) Lesson 3: Pebble in Pond Wave Equation
# - Unit 8 (OpenACC) Lesson 1: Accelerating Scientific Applications
# - Unit 10 (Productivity and Visualization) Lesson 3: Visualization 1
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

.PHONY: clean all all1 test grey

all: all1

clean:
	rm -rf *.o ${THE_LIBRARY} ${TESTS} jpeg_library.stamp
	(cd jpeg-6b; make clean)

#
# Where to get PNG from
#
PNG_CFLAGS = ${shell pkg-config --cflags libpng}
PNG_LIBS = ${shell pkg-config --libs libpng}

#
# Where to get JPEG from
#
JPEG_TOP = ./jpeg-6b
JPEG_CFLAGS = -I${JPEG_TOP}
JPEG_LIBS = -L${JPEG_TOP} -ljpeg

CFLAGS = -g ${PNG_CFLAGS} ${JPEG_CFLAGS}
LIBS = ${PNG_LIBS} ${JPEG_LIBS} -lm

THE_LIBRARY = libmimage.a

LIB_C_SRCS = misc.c mimage.c mpng.c mjpeg.c
LIB_C_OBJS = ${LIB_C_SRCS:.c=.o}

${THE_LIBRARY}: jpeg_library.stamp ${LIB_C_OBJS}
	${AR} cr ${THE_LIBRARY} ${LIB_C_OBJS}

TESTS = t1 trpng trjpeg tmandelbrot ttest tpebOscalar tpebOmp tpebOmpi tpebOmpimp tpebO tpebA tpebM tpebble tgrey tframe
GREYS = t1 trpng trjpeg tgrey tframe


t1: t1.o ${THE_LIBRARY}
	${CC} -o t1 t1.o ${THE_LIBRARY} ${LIBS}


trpng: trpng.o ${THE_LIBRARY}
	${CC} -o trpng trpng.o ${THE_LIBRARY} ${LIBS}

trjpeg: trjpeg.o ${THE_LIBRARY}
	${CC} -o trjpeg trjpeg.o ${THE_LIBRARY} ${LIBS}

ttest: ttest.o ${THE_LIBRARY}
	${CC} -o ttest ttest.o ${THE_LIBRARY} ${LIBS}

tgrey: tgrey.o ${THE_LIBRARY}
	${CC} -o tgrey tgrey.o ${THE_LIBRARY} ${LIBS}

tframe: tframe.o ${THE_LIBRARY}
	${CC} -o tframe tframe.o ${THE_LIBRARY} ${LIBS}

tpebble: tpebble.o ${THE_LIBRARY}
	${CC} -o tpebble tpebble.o ${THE_LIBRARY} ${LIBS}

tpebM: tpebM.o ${THE_LIBRARY}
	${CC} -o tpebM tpebM.o ${THE_LIBRARY} ${LIBS}

tpebA: tpebA.o ${THE_LIBRARY}
	${CC} -o tpebA tpebA.o ${THE_LIBRARY} ${LIBS}

tpebO: tpebO.o ${THE_LIBRARY}
	${CC} -o tpebO tpebO.o ${THE_LIBRARY} ${LIBS}

tpebOmpimp: tpebOmpimp.o ${THE_LIBRARY}
	${CC} -o tpebOmpimp tpebOmpimp.o ${THE_LIBRARY} ${LIBS}

tpebOmpi: tpebOmpi.o ${THE_LIBRARY}
	${CC} -o tpebOmpi tpebOmpi.o ${THE_LIBRARY} ${LIBS}

tpebOmp: tpebOmp.o ${THE_LIBRARY}
	${CC} -o tpebOmp tpebOmp.o ${THE_LIBRARY} ${LIBS}

tpebOscalar: tpebOscalar.o ${THE_LIBRARY}
	${CC} -o tpebOscalar tpebOscalar.o ${THE_LIBRARY} ${LIBS}




tmandelbrot: tmandelbrot.o ${THE_LIBRARY}
	${CC} -o tmandelbrot tmandelbrot.o ${THE_LIBRARY} ${LIBS}

all1: ${THE_LIBRARY} ${TESTS} ${GREYS}

test: all
	./tframe ; display ../testImageMagik/data/gray_frame00000.png
	./tgrey ; display tgrey00001.png
	./trpng examples/cluster.png
	./trjpeg examples/testimg.jpg
	./tmandelbrot ; display tmand.png
	./ttest ; display ttest.png
bluewaters: all
	./tpebble ; display ../Vis-Movie/data/pebble_mvlp00020.png
	./tpebM ; display ../MPI-Waves/data/pebble_mvlp00020.png
	./tpebA ; display ../OpenACC-Pebble/data/pebble_mvlp00020.png
	./tpebOmpimp ; display ../MPI-OpenMP-Pebble/openmpimp/data/pebble_mvlp00020.png
	./tpebOmpi ; display ../MPI-OpenMP-Pebble/openmpi/data/pebble_mvlp00020.png
	./tpebOmp ; display ../MPI-OpenMP-Pebble/openmp/data/pebble_mvlp00020.png
	./tpebOscalar ; display ../MPI-OpenMP-Pebble/scalar/data/pebble_mvlp00020.png

grey: all
	./tgrey ; display tgrey00001.png
	./tframe ; display tframe00001.png


jpeg_library.stamp:
	(cd jpeg-6b; ./configure; make)
	touch jpeg_library.stamp
