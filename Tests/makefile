CFLAGS=-I..


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


all: test1 test2 test3

FRACDOBJ = $(patsubst %.c,%.o,$(wildcard ../*.c))
test1a: 
	@echo $(FRACDOBJ) 


test1: test1.o ${FRACDOBJ} chkopts
	-@${CLINKER} -o test1 test1.o ${FRACDOBJ} ${PETSC_LIB}
	-rm -f $(FRACDOBJ) 

test2: test2.o ${FRACDOBJ} chkopts
	-@${CLINKER} -o test2 test2.o ${FRACDOBJ} ${PETSC_LIB}
	-rm -f $(FRACDOBJ) 

test3: test3.o ${FRACDOBJ} chkopts
	-@${CLINKER} -o test3 test3.o ${FRACDOBJ} ${PETSC_LIB}
	-rm -f $(FRACDOBJ) test3.o

clean::
	-rm -f test1 test2 test3 *.vtk $(FRACDOBJ) 