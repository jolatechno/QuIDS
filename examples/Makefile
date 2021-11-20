U_CFLAGS=$(CFLAGS) -O3 -fopenmp --std=c++2a

targets=$(basename $(wildcard **.cpp))

all: $(targets)

clean:
	rm *.out

$(targets):
	$(CXX) $@.cpp -o$@.out $(U_CFLAGS)