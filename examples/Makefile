U_CFLAGS=$(CFLAGS) --std=c++2a -O3 -fopenmp

targets=$(basename $(wildcard **.cpp))

all: $(targets)

clean:
	rm *.out

$(targets):
	$(CXX) $@.cpp -o$@.out $(U_CFLAGS) 