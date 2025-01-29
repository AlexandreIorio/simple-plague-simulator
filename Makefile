.PHONY: all std omp cuda clean display_timeline generate_video timeline_details

MAKEFILE_DIR = makefiles
all: std omp cuda

std:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.std

omp:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.omp

cuda:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.cuda

display_timeline:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.display

generate_video:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.video

timeline_details:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.details

clean:
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.std clean
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.omp clean
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.cuda clean
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.display clean
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.video clean
	$(MAKE) -f $(MAKEFILE_DIR)/Makefile.details clean
