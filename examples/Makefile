U_CFLAGS=$(CFLAGS) -O3 -fopenmp -ltbb

targets=$(basename $(wildcard **.cpp))

all: $(targets)

clean:
	rm *.out

$(targets):
	$(CXX) $@.cpp -o$@.out $(U_CFLAGS)