.PHONY: all std omp cuda clean

all: std omp cuda

std:
	$(MAKE) -f Makefile.std

omp:
	$(MAKE) -f Makefile.omp

cuda:
	$(MAKE) -f Makefile.cuda

clean:
	$(MAKE) -f Makefile.std clean
	$(MAKE) -f Makefile.omp clean
	$(MAKE) -f Makefile.cuda clean
