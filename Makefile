CXX = g++
CFLAGS = -O3 -g -funroll-loops -ftree-vectorize -fopenmp -mfma -mavx2 -mavx512f -march=native -pthread -std=c++11
LIBS = -lllib -lntl -lgmp -lm -lgf2x
LIBDIR = -I$(HOME)/work/packages/include/llib -L$(HOME)/work/packages/lib

all: test

test: test.cpp $(OBJ)
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(LIBS) -o test

split_gaussian: split_gaussian.cpp $(OBJ)
	$(CXX) $(CFLAGS) $^ $(LIBDIR) $(LIBS) -o split_gaussian

clean:
	-rm $(ASMOBJ) $(OBJ) test
