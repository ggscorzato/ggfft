# this is the makefile for the thimble_monte_carlo code.

# system specific definitions:

SRCDIR=.
OBJDIR=.
BINDIR=.
MPIDIR=
CC=mpicc -g
AR=ar

# code specific definitions:

TGLIB = libggfft.a
TGPROG = test_fft
CFLAGS=-I. -g -msse3 -Wall
LDFLAGS=-L. -lggfft  -lm -Wall

# general definitions and rules:  FIX THE LACK O FDEPENDENCY ON HEADERS!!!!

#DEPFLAGS = -MM
SOURCES := $(wildcard $(SRCDIR)/*.c)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
#DEPS := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.d)
rm = rm -f

all: $(TGLIB) $(TGPROG)

$(BINDIR)/$(TGLIB) : $(OBJECTS)
	@$(AR) cr $(BINDIR)/$(TGLIB) $(OBJECTS)
	@echo "Library complete!"

$(BINDIR)/$(TGPROG) : $(OBJECTS)
	@$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

.PHONEY: clean
clean:
	@$(rm) $(OBJECTS) 
	@echo "Cleanup complete!"

.PHONEY: rm
remove: 
	@$(rm) $(OBJECTS) $(BINDIR)/$(TGPROG) $(BINDIR)/$(TGLIB)
	@echo "Executable removed!"
