U_CXX=$(CXX) -O3 -std=c++2a
U_CFLAGS=$(CFLAGS) -fopenmp -ltbb

targets=$(basename $(wildcard **.cpp))

all: $(targets)

clean:
	rm *.out

$(targets):
	$(U_CXX) $@.cpp -o$@.out $(U_CFLAGS)