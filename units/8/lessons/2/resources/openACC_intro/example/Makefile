# Makefile
#------- Select your Compier
#COMPTYPE="Cray"
#COMPTYPE="Intel"
#COMPTYPE="GNU"
COMPTYPE="PGI"

#------ Select known target machine
#SYSTYPE="Earlham"
#SYSTYPE="Stampede"
SYSTYPE="BW"
#SYSTYPE="Mac"
#SYSTYPE="Linux"

ifeq ($(SYSTYPE), "Earlham")
CC			= pgcc
MPICC		= mpicc
OMPFLAGS	= -openmp
ACCFLAGS	= -acc -Minfo
LIBS		= -lm
endif

ifeq ($(SYSTYPE), "BW")
    ifeq ($(COMPTYPE), "Cray")
    CC			= cc
    ACCFLAGS    = -h pragma=acc,msgs
    MPICC		= cc
    OMPFLAGS	= -fopenmp
    LIBS		= -lm
	else ifeq ($(COMPTYPE), "PGI")
    CC          = pgcc
    MPICC       = mpicc 
    ACCFLAGS    = -acc -Minfo
    OMPFLAGS    = -mp=nonuma
    LIBS        = -lm
    else ifeq ($(COMPTYPE), "GNU")
    CC          = gcc
    MPICC       = mpicc 
    ACCFLAGS    = -fopenaccc
    OMPFLAGS    = -fopenmp
    LIBS        = -lm
	endif
endif

laplace_serial:
	$(CC) $(ACC) -o laplace_serial.exe laplace_serial.c   
laplace_bad_acc:
	$(CC) $(ACC) -o laplace_bad_acc.exe laplace_bad_acc.c
laplace_acc:
	$(CC) $(ACC) -o laplace_acc.exe laplace_acc.c
vect_add:
	$(CC) $(ACC) -o vect_add.exe vect_add.c
all:
	make clean
	make laplace_serial laplace_bad_acc vect_add
clean:
	rm -rf *.exe *.ptx *.cub
