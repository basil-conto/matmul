# ------------------------------------------------------------------------------
# Makefile for the matmul matrix multiplication program using Open MPI.
#
# Copyright (c) 2015, Basil L. Contovounesios <contovob@tcd.ie>
#
# This code is distributed under the BSD3 license. See LICENSE.
# ------------------------------------------------------------------------------

# Open MPI compiler
CC := mpicc

# Target, source and optimisation flags
prog_OUT := matmul
prog_SRC := $(prog_OUT).c
prog_OPT := -march=native -funroll-loops -flto -O3

# Append to preexisting CFLAGS, whether those are defined by the environment or
# at the command line
override CFLAGS += -std=gnu11 $(prog_OPT)

.PHONY: all clean

all: $(prog_OUT)

$(prog_OUT): $(prog_SRC)
	$(LINK.c) -o $@ $<

clean:
	@- $(RM) $(prog_OUT)
