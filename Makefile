.PHONY: all std openmp cuda clean

all: std openmp cuda

std:
	$(MAKE) -f Makefile.std

openmp:
	$(MAKE) -f Makefile.openmp

cuda:
	$(MAKE) -f Makefile.cuda

clean:
	$(MAKE) -f Makefile.std clean
	$(MAKE) -f Makefile.openmp clean
	$(MAKE) -f Makefile.cuda clean
