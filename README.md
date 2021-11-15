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
	iqs::simulate(state, [](char *parent_begin, char *parent_end, std::complex<PROBA_TYPE> &mag) {
			/* using lambda-expressions */
		});

	/* applying a rule */
	iqs::rule_t *rule = new my_rule(/*...*/);
	iqs::simulate(state, rule, buffer, symbolic_iteration);
}
```

### MPI support

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
	iqs::simulate(state, [](char *parent_begin, char *parent_end, std::complex<PROBA_TYPE> &mag) {
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
void my_modifier(char *parent_begin, char *parent_end, std::complex<PROBA_TYPE> &mag) {
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

	inline void populate_child(char const *parent_begin, char const *parent_end,
		char* const child_begin, uint32_t const child_id,
		size_t &size, std::complex<PROBA_TYPE> &mag) const override;

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
inline void populate_child(char const *parent_begin, char const *parent_end,
	char* const child_begin, uint32_t const child_id,
	size_t &size, std::complex<PROBA_TYPE> &mag) const override
{
	// modify mag...
	// populate the child, starting at child_begin
	size = actual_child_size;
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

	void append(char const *object_begin_, char const *object_end_, std::complex<PROBA_TYPE> const mag=1);
	void pop(uint n=1, bool normalize_=true);
	void get_object(size_t const object_id, char *& object_begin, size_t &object_size, std::complex<PROBA_TYPE> *&mag);
	void get_object(size_t const object_id, char const *& object_begin, size_t &object_size, std::complex<PROBA_TYPE> &mag) const;

	template<class T>
	T average_value(std::function<T(char const *object_begin, char const *object_end)> const &observable) const;

private:
	/*...*/
};
```

The `iteration` class (or `it_t` type) has two constructors, a basic one, and one that simply takes a starting object and append it to the state with probability one.

Member functions are:
- `append(...)` : Append an object to the state, with a give magnitude (default = 1).
- `pop(...)` : Remove the `n` last objects, and normalze (if `normalize_` is `true`).
- `get_object(...)` : Allows to read (either as constant or not) an objects and its magnitude, with a given `object_id` between 0 and `num_object`. Note that the non-constant function takes pointers for `mag`.
- `average_value(...)` : Compute the average value of an observable (a function) of any type (that can be added, initialized by `T x = 0`, and multiplied by an object of type `PROBA_TYPE`).

Member variables are:
- `num_object` : Number of object describing this state currently in superposition.
- `total_proba` : total probability held by this state before normalizing it (so after truncation).

#### MPI symbolic iteration

```cpp
typedef class mpi_symbolic_iteration mpi_sy_it_t;

class mpi_symbolic_iteration : public iqs::symbolic_iteration {
public:
	size_t get_total_num_object(MPI_Comm communicator) const;
	size_t get_total_num_object_after_interferences(MPI_Comm communicator) const;
	mpi_symbolic_iteration() {}

private:
	/*...*/
};
```

The `mpi_symbolic_iteration` class (or `mpi_sy_it_t` type) can be considered as exactly equivalent to the `symbolic_iteration` class (or `sy_it_t` type).

The two added member functions are:
- `get_total_num_object(...)` : Get the total number of object at symbolic iteration accross all nodes.
- `get_total_num_object_after_interferences(...)` : Get the total number of object at symbolic iteration, after interferences but before truncation, accross all nodes.

#### MPI iteration

```cpp
typedef class mpi_iteration mpi_it_t;

class mpi_iteration : public iqs::iteration {
public:
	PROBA_TYPE node_total_proba = 0;

	mpi_iteration() {}
	mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

	void equalize(MPI_Comm communicator);
	size_t get_total_num_object(MPI_Comm communicator) const;
	void send_objects(size_t num_object_sent, int node, MPI_Comm communicator);
	void receive_objects(int node, MPI_Comm communicator);
	void distribute_objects(MPI_Comm comunicator, int node_id);
	void gather_objects(MPI_Comm comunicator, int node_id);

	template<class T>
	T average_value(std::function<T(char const *object_begin, char const *object_end)> const &observable, MPI_Comm communicator) const

private:
	/*...*/
};
```

The `mpi_iteration` class (or `mpi_it_t` type) inehrits all the public memeber functions and varibale of the `iteration` class (or `it_t` type), and shares similar constructors.

The additional member functions are:
- `equalize(...)` : Does its best at equalizing the number of object on each node. Will only equalize among pair (in hopefully the optimal pair-arangment), so it's up to you to check if the objects are equally shared among nodes, as some spetial cases can't be equalized well by this algorithm. `normalize(MPI_Comm ...)` should be after `equalize(...)` at the end to compute `node_total_proba`.
- `get_total_num_object(...)` : Get the total number of object accross all nodes.
- `send_objects(...)` : Send a given number of object to a node, and `pop` them of the sending one. `normalize(MPI_Comm ...)` should be after `send_objects(...)` at the end to compute `node_total_proba`.
- `receive_objects(...)` : Receiving end of the `send_objects(...)` function. `normalize(MPI_Comm ...)` should be after `receive_objects(...)` at the end to compute `node_total_proba`.
- `distribute_objects(..)` : Distribute objects that are located on a single node of id `node_id` (0 if not specified) equally on all other nodes. `normalize(MPI_Comm ...)` should be after `distribute_objects(...)` at the end to compute `node_total_proba`.
- `gather_objects(...)` : Gather objects on all nodes to the node of id `node_id` (0 if not specified). If all objects can't fit on the memory of this node, the function will throw a `bad alloc` error as the behavior is undefined. `node_total_proba` is calculated at the end as it doesn't require a calling `normalize(MPI_Comm ...)`.
- `average_value(...)` : equiavlent to the normal `iteration` member function, but for the whole distributed wave function (__note that calling__ `average_value(...)` __without an__ `MPI_Comm` __will return a local average value for retrocompatibility with the basic__ `iteration` __class__).

`node_total_proba` is the only additional member variable, and is the proportion of total probability that is held by a given node.

### Global parameters

In addition to classes, some global parameters are used to modify the behaviour of the simulation.

```cpp
namespace iqs {
	PROBA_TYPE tolerance = TOLERANCE;
	float safety_margin = SAFETY_MARGIN;
	float collision_test_proportion = COLLISION_TEST_PROPORTION;
	float collision_tolerance = default is COLLISION_TOLERANCE;

	namespace mpi {
		size_t min_equalize_size = MIN_EQUALIZE_SIZE;
		float equalize_imablance = EQUALIZE_IMBALANCE;

		/* ... */
	}

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

`tolerance` represents the minimum probability considered non-zero (default is `1e-18`, to compensate for numerical errors).

#### safety margin

`safety_margin` represents the target proportion of memory to keep free (default is `0.2` for 20%).

#### collision test proportion and collision tolerance

`collision_test_proportion` represents the proportion of objects for which with first remove duplicates, we then continue removing duplicates only if the proportion of duplicates is greater than `collision_tolerance`.

`collision_test_proportion` has a default of  `0.1` and `collision_tolerance` has a default of `0.05`.

#### minimum equalize size and equalize imbalance.

`mpi::min_equalize_size` represents the minimum per node average size required to automaticly call `equalize(...)` after a call to `iqs::mpi::simulate(...)`.

If this first condition is met, `equalize(...)` if the maximum relative imbalance in the number of object accross the nodes is greater than `mpi::equalize_imablance`.

`mpi::min_equalize_size` has the same default as `utils::min_vector_size` by default, and `mpi::equalize_imablance` has a default of `0.2`.

#### min vector size

`utils::min_vector_size` represent the minimum size of any vector (the default is `100000`).

#### upsize policy

`utils::upsize_policy` represent the multiplier applied when upsizing a vector (the default is `1.1`). It avoid frequent upsizing by giving a small margin.

#### downsize policy

`utils::downsize_policy` reprensent the threshold multiplier to downsize a vector (the default is `0.85`). A vector won't be downsized until the requested size is smaller than this ultiplier times the capacity of the given vector.

__!! this multiplier should always be smaller than the inverse of upsize_policy to avoid upsizing-downsizing loop !!__
