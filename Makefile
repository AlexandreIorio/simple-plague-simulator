.PHONY: all std omp cuda clean display_timeline generate_video

all: std omp cuda

std:
	$(MAKE) -f Makefile.std

omp:
	$(MAKE) -f Makefile.omp

cuda:
	$(MAKE) -f Makefile.cuda

display_timeline:
	$(MAKE) -f Makefile.display

generate_video:
	$(MAKE) -f Makefile.video

clean:
	$(MAKE) -f Makefile.std clean
	$(MAKE) -f Makefile.omp clean
	$(MAKE) -f Makefile.cuda clean
	$(MAKE) -f Makefile.display clean
	$(MAKE) -f Makefile.video clean
