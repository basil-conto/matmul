# ------------------------------------------------------------------------------
# Makefile for the Matrix.java matrix library.
#
# Copyright (c) 2015, Basil L. Contovounesios <contovob@tcd.ie>
#
# This code is distributed under the BSD3 license. See LICENSE.
# ------------------------------------------------------------------------------

JC  := javac
JVM := java

prog_NAME := Benchmark
prog_SRCS := $(wildcard *.java)
prog_OBJS := $(prog_SRCS:.java=.class)

override JFLAGS   += -O
# override JVMFLAGS += -ea

.PHONY: all run $(prog_NAME) clean

# Define .java - .class dependency
.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

all: $(prog_NAME)

# Run program
run: $(prog_NAME)
	$(JVM) $(JVMFLAGS) $< $(ARGS)

# Compile program
$(prog_NAME): $(prog_OBJS)

clean:
	@- $(RM) *.class
