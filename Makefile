.PHONY: all src lib test clean

all:
	make clean
	(cd src; make)
	(cd lib; make)
	(cd test; make)
#	(cd kernel_test; make)
src:
	(cd src; make)

lib:
	(cd lib; make)

test:
	(cd test; make)

#kernel_test:
#	(cd kernel_test; make)

clean:
	(cd src; make clean)
	(cd lib; make clean)
	(cd test; make clean)
	(cd kernel_test; make clean)
