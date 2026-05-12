# Makefile for neural-plato Fortran modules
# Compiles all Fortran sources into object files for linking with Rust

FC = gfortran
FFLAGS = -O2 -std=f2008 -fPIC -Wall -Wextra -fcheck=all
MOD_DIR = .

SRCS = sparse_memory.f90 amnesia_curve.f90 intent_snap.f90 tucker_decompose.f90 negative_space.f90 neural_plato.f90
OBJS = $(SRCS:.f90=.o)

.PHONY: all clean lib example

all: $(OBJS) lib

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@ -J$(MOD_DIR)

lib: $(OBJS)
	$(FC) -shared -o libneural_plato.so $(OBJS) -lm

example: basic.f90 $(OBJS)
	$(FC) $(FFLAGS) -o basic_example basic.f90 $(OBJS) -lm -I$(MOD_DIR)

clean:
	rm -f *.o *.mod libneural_plato.so basic_example
