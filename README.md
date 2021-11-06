# Irregular Quantum Simulator

## Installation

This library is header-only, so you can simply link files in [src](./src).

### Requirements

The only requirement is to have at least `c++17`, and `tbb` ([Thread Building Blocks](https://github.com/oneapi-src/oneTBB)). Parallelism is implemented using `OpenMP`.

`onetbb` can be installed using the [tbb-install.sh](./tbb-install.sh), with the flags to pass to make being outputed by [get-tbb-cflags.sh](./get-tbb-cflags.sh).

## Usage

Objects are represented by a simple begin and end pointer. Their exist two kind of interfaces for implementing a unitary transformation.

`modifiers` and `rules` are applied using the `iqs::simulate(...)` function:

```cpp
#include "src/iqs.hpp"

int main(int argc, char* argv[]) {
	/* variables*/
	iqs::it_t buffer, state;
	iqs::sy_it_t symbolic_iteration;

	/* initializing the state */
	state.append(object_begin, object_end);

	/* applying a modifier */
	iqs::simulate(state, my_modifier);
	iqs::simulate(state, [](char *parent_begin, char *parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			/* using lambda-expressions */
		});

	/* applying a rule */
	iqs::rule_t *rule = new my_rule(/*...*/);
	iqs::simulate(state, rule, buffer, symbolic_iteration);
}
```

### MPI support

__!!! MPI support is currently being added to this branch, it will be merged with the main branch, and a new release will be created when MPI is fully supported !!!__

Simulations can also be done across nodes. For that, you'll need to replace `iqs::sy_it` and `iqs::it_t` respectivly by `iqs::mpi::mpi_sy_it` and `iqs::mpi::mpi_it_t`. 

```cpp
#include "src/iqs_mpi.hpp"

int main(int argc, char* argv[]) {
	/* MPI initialization */
	int size, rank;
	MPI_Init(&argc, &argv);

	/* variables*/
	iqs::mpi::mpi_it_t buffer, state;
	iqs::mpi::mpi_sy_it_t symbolic_iteration;

	/* initializing the state */
	state.append(object_begin, object_end);

	/* applying a modifier */
	iqs::simulate(state, my_modifier);
	iqs::simulate(state, [](char *parent_begin, char *parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			/* using lambda-expressions */
		});

	/* applying a rule */
	iqs::rule_t *rule = new my_rule(/*...*/);
	iqs::mpi::simulate(state, rule, buffer, symbolic_iteration, MPI_COMM_WORLD);

	MPI_Finalize();
}
```

### Modifiers

A `modifier` is a simple functions that takes a objects, and modify it in place, while keep its size unchanged.

```cpp
void my_modifier(char *parent_begin, char *parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
	// modify the object...
}
```

### Rules

A `rule` is a simple class, implementing 2 functions (with a third being optional).

```cpp
class my_rule : public iqs::rule_t {
public:
	my_rule() {};
	inline void get_num_child(char const *parent_begin, char const *parent_end, 
		uint32_t &num_child, size_t &max_child_size) const override;

	inline char* populate_child(char const *parent_begin, char const *parent_end, 
		uint32_t child_id, 
		PROBA_TYPE &real, PROBA_TYPE &imag,
		char* child_begin) const override;

	inline size_t hasher(char const *parent_begin, char const *parent_end) const; // optional
};
```

The first function, `get_num_child(...)`, finds the number of objects created through the unitary transform by a given objects. It also gives an upper-bound to the size of those objects. 

```cpp
inline void my_rule::get_num_child(char const *parent_begin, char const *parent_end,
	uint32_t &num_child, size_t &max_child_size) const override
{
	// do stuff...
	num_child = actual_num_child;
	max_child_size = actual_max_child_size;
}
```

The second function, `populate_child(...)`, simply "populate" (create) an object from its parent, while also modifying its magnitude according to the unitary transformation (the starting magnitude is the parent magnitude).

Note that `child_id` is a number from `0` to `num_child`, and simply identify "child" amoung its "siblings" (objects with the same "parent").

```cpp
inline char* my_rule::populate_child(char const *parent_begin, char const *parent_end,
	uint32_t child_id,
	PROBA_TYPE &real, PROBA_TYPE &imag,
	char* child_begin) const override
{
	// modify imag and real...
	// populate the child, starting at child_begin
	return child_end;
}
```

The last function is a hasher for objects. If not specified, the whole memory buffer will simply be hashed. A hasher NEEDS to be provided if objects that are equal can be represented by different objects. The default implementation is:

```cpp
inline size_t my_rule::hasher(char const *parent_begin, char const *parent_end) const override {
	return std::hash<std::string_view>()(
		std::string_view(parent_begin, std::distance(parent_begin, parent_end)));
}
```

### Interaction with the different classes

We can see that a quantum state is represented by a specific `iteration` class, and a `symbolic_iteration` is generated when applying a unitary transformation on a state. We interact with those classes (modify or read a state, ect...) through public member functions and variables, which will be shortly documented bellow, as they are vital to building a __usefull__ program using `IQS`.

#### Symbolic iteration

```cpp
typedef class symbolic_iteration sy_it_t;

class symbolic_iteration {
public:
	size_t num_object = 0;
	size_t num_object_after_interferences = 0;

	symbolic_iteration() {}

private:
	/*...*/
};
```

The `symbolic_iteration` class (or `sy_it_t` type) only has a basic constructor, as it is only ment to be used internaly.

Member variables are:
- `num_object` : Number of objects generated at symbolic iteration (before interferences and truncation).
- `num_object_after_interferences` : Number of objects after eliminating duplicates (so called "interferences"), but before truncation.

#### Iteration

```cpp
typedef class iteration it_t;

class iteration {
public:
	size_t num_object = 0;
	PROBA_TYPE total_proba = 1;

	iteration();
	iteration(char* object_begin_, char* object_end_);

	void append(char* object_begin_, char* object_end_, PROBA_TYPE real_ = 1, PROBA_TYPE imag_ = 0);
	char* get_object(size_t object_id, size_t &object_size, PROBA_TYPE *&real_, PROBA_TYPE *&imag_);
	char const* get_object(size_t object_id, size_t &object_size, PROBA_TYPE &real_, PROBA_TYPE &imag_) const;

private:
	/*...*/
};
```

The `iteration` class (or `it_t` type) has two constructors, a basic one, and one that simply takes a starting object and append it to the state with probability one.

Member functions are:
- `append(...)` : Append an object to the state, with a give magnitude (default = 1).
- `get_object(...)` : Allows to read (either as constant or not) an objects and its magnitude, with a given `object_id` between 0 and `num_object`. Note that the non-constant function takes pointers for `real` and `imag`.

Member variables are:
- `num_object` : Number of object describing this state currently in superposition.
- `total_proba` : total probability held by this state before normalizing it (so after truncation).

#### MPI symbolic iteration

```cpp
typedef class mpi_symbolic_iteration mpi_sy_it_t;

class mpi_symbolic_iteration : public iqs::symbolic_iteration {
public:
	mpi_symbolic_iteration() {}

private:
	/*...*/
};
```

The `mpi_symbolic_iteration` class (or `mpi_sy_it_t` type) can be considered as exactly equivalent to the `symbolic_iteration` class (or `sy_it_t` type).

#### MPI iteration

```cpp
typedef class mpi_iteration mpi_it_t;

class mpi_iteration : public iqs::iteration {
public:
	mpi_iteration() {}
	mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

	void distribute_objects(MPI_Comm comunicator, int node_id);
	void gather_objects(MPI_Comm comunicator, int node_id);

private:
	/*...*/
};
```

The `mpi_iteration` class (or `mpi_it_t` type) inehrits all the public memeber functions and varibale of the `iteration` class (or `it_t` type), and shares similar constructors.

The additional member functions are:
- `distribute_objects(..)` : distribute objects that are located on a single node of id `node_id` (0 if not specified) equally on all other nodes.
- `gather_objects(...)` : gather objects on all nodes to the node of id `node_id` (0 if not specified). If all objects can't fit on the memory of this node, the function will throw a `bad alloc` error as the behavior is undefined.

### Global parameters

In addition to classes, some global parameters are used to modify the behaviour of the simulation.

```cpp
namespace iqs {
	void set_tolerance(PROBA_TYPE val) { /* ... */ } // default is TOLERANCE
	void set_safety_margin(float val) { /* ... */ } // default is SAFETY_MARGIN
	void set_collision_test_proportion(float val) { /* ... */ } // default is COLLISION_TEST_PROPORTION
	void set_collision_tolerance(float val) { /* ... */ } // default is COLLISION_TOLERANCE

	namespace utils {
		float upsize_policy = UPSIZE_POLICY;
		float downsize_policy = DOWNSIZE_POLICY;
		size_t min_vector_size = MIN_VECTOR_SIZE;

		/* ... */
	}

	/* ... */
}
```

The default value of any of those variable can be altered at compilation, by passing an uppercase flag with the same name as the desired variable.

#### tolerance

`tolerance` (which is only accessible through `set_tolerance()`) represents the minimum probability considered non-zero (default is `1e-18`, to compensate for numerical errors).

#### safety margin

`safety_margin` (which is only accessible through `set_safety_margin()`) represents the target proportion of memory to keep free (default is `0.2` for 20%).

#### collision test proportion and collision tolerance

`collision_test_proportion` (which is only accessible through `set_safety_margin()`) represents the proportion of objects for which with first remove duplicates, we then continue removing duplicates only if the proportion of duplicates is greater than `collision_tolerance` (which is itself only accessible through `set_collision_tolerance()`).

`collision_test_proportion` has a default of  `0.1` and `collision_tolerance` has a default of `0.05`.

#### min vector size

`utils::min_vector_size` represent the minimum size of any vector (the default is `100000`).

#### upsize policy

`utils::upsize_policy` represent the multiplier applied when upsizing a vector (the default is `1.1`). It avoid frequent upsizing by giving a small margin.

#### downsize policy

`utils::downsize_policy` reprensent the threshold multiplier to downsize a vector (the default is `0.85`). A vector won't be downsized until the requested size is smaller than this ultiplier times the capacity of the given vector.

__!! this multiplier should always be smaller than the inverse of upsize_policy to avoid upsizing-downsizing loop !!__

## TODOS

- implementing (basic) `MPI` support.
- implementing a third kind of unitary transform, discribing a transformation that doesn't change the size of an object (removing the need for a symbolic iteration).