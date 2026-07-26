include ../make.inc

all: libpangulu.a


libpangulu.a:
	ar -rv -o $@ ../src/pangulu.o 
	- ranlib $@
clean:
	rm -f libpangulu.a
