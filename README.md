# Quantum Irregular Dynamic Simulator 

## Installation

This library is header-only, so you can simply link files in [src](./src).

### Requirements

The only requirement is to have at least `c++2a`. Parallelism is implemented using `OpenMP`, although the pure `MPI` implementation is more efficient right now.

## Documentation

The code is documented using `doxygen`. Documentation is present at [doc/](./doc/) and is also hosted at [codedocs.xyz/jolatechno/QuIDS](https://codedocs.xyz/jolatechno/QuIDS).

## Usage
Some rules that can be used directly, or understood as examples are implemented in [src/rules](./src/rules).

Objects are represented by a simple begin and end pointer. Their exist two kind of interfaces for implementing a unitary transformation.

`modifiers` and `rules` are applied using the `quids::simulate(...)` function:

```cpp
#include "src/quids.hpp"

int main(int argc, char* argv[]) {
	/* variables*/
	quids::it_t next_state, state;
	quids::sy_it_t symbolic_iteration;

	/* initializing the state */
	state.append(object_begin, object_end);

	/* applying a modifier */
	quids::simulate(state, my_modifier);
	quids::simulate(state, [](char *parent_begin, char *parent_end, std::complex<PROBA_TYPE> &mag) {
			/* using lambda-expressions */
		});

	/* applying a rule */
	quids::rule_t *rule = new my_rule(/*...*/);
	quids::simulate(state, rule, next_state, symbolic_iteration);

	/* "next_state" now holds an application of "rule" on "state" */
}
```

### MPI support

Simulations can also be done across nodes. For that, you'll need to replace `quids::sy_it` and `quids::it_t` respectivly by `quids::mpi::mpi_sy_it` and `quids::mpi::mpi_it_t`. 

```cpp
#include "src/quids_mpi.hpp"

int main(int argc, char* argv[]) {
	/* MPI initialization */
	int size, rank, provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if(provided < MPI_THREAD_SERIALIZED) {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

	/* variables*/
	quids::mpi::mpi_it_t next_state, state;
	quids::mpi::mpi_sy_it_t symbolic_iteration;

	/* initializing the state */
	state.append(object_begin, object_end);

	/* applying a modifier */
	quids::simulate(state, my_modifier);
	quids::simulate(state, [](char *parent_begin, char *parent_end, std::complex<PROBA_TYPE> &mag) {
			/* using lambda-expressions */
		});

	/* applying a rule */
	quids::rule_t *rule = new my_rule(/*...*/);
	quids::mpi::simulate(state, rule, next_state, symbolic_iteration, MPI_COMM_WORLD);

	/* "next_state" now holds an application of "rule" on "state" */

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
class my_rule : public quids::rule_t {
public:
	my_rule() {};
	inline void get_num_child(char const *parent_begin, char const *parent_end, 
		size_t &num_child, size_t &max_child_size) const override;

	inline void populate_child(char const *parent_begin, char const *parent_end,
		char* const child_begin, uint32_t const child_id,
		size_t &size, std::complex<PROBA_TYPE> &mag) const override;

	inline void populate_child_simple(char const *parent_begin, char const *parent_end,
		char* const child_begin, uint32_t const child_id) const; // optional

	inline size_t hasher(char const *parent_begin, char const *parent_end) const; // optional
};
```

The first function, `get_num_child(...)`, finds the number of objects created through the unitary transform by a given objects. It also gives an upper-bound to the size of those objects. 

```cpp
inline void my_rule::get_num_child(char const *parent_begin, char const *parent_end,
	size_t &num_child, size_t &max_child_size) const override
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

The third function is `populate_child_simple(...)` which is simply a copy of `populate_child(...)` that can skip the computation of the magnitude and the size of the child objects. If not provided the default implementation relize on `populate_child(...)`:

```cpp
inline void populate_child_simple(char const *parent_begin, char const *parent_end,
	char* const child_begin, uint32_t const child_id) const { //can be overwritten
		size_t size_placeholder;
		mag_t mag_placeholder;
		populate_child(parent_begin, parent_end, child_begin, child_id,
			size_placeholder, mag_placeholder);
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

We can see that a quantum state is represented by a specific `iteration` class, and a `symbolic_iteration` is generated when applying a unitary transformation on a state. We interact with those classes (modify or read a state, ect...) through public member functions and variables, which will be shortly documented bellow, as they are vital to building a __usefull__ program using `QuIDS`.

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

class mpi_symbolic_iteration : public quids::symbolic_iteration {
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

class mpi_iteration : public quids::iteration {
public:
	PROBA_TYPE node_total_proba = 0;

	mpi_iteration() {}
	mpi_iteration(char* object_begin_, char* object_end_) : quids::iteration(object_begin_, object_end_) {}

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

### Global parameters and pre-processor flags

In addition to classes, some global parameters are used to modify the behaviour of the simulation.

```cpp
#ifndef PROBA_TYPE
	#define PROBA_TYPE double
#endif
#ifndef HASH_MAP_OVERHEAD
	#define HASH_MAP_OVERHEAD 1.5
#endif

/* other default-value flags */

namespace quids {
	PROBA_TYPE tolerance = TOLERANCE;
	float safety_margin = SAFETY_MARGIN;
	int load_balancing_bucket_per_thread = LOAD_BALANCING_BUCKET_PER_THREAD;
	#ifdef SIMPLE_TRUNCATION
		bool simple_truncation = true;
	#else
		bool simple_truncation = false;
	#endif

	namespace mpi {
		size_t min_equalize_size = MIN_EQUALIZE_SIZE;
		float equalize_inbalance = EQUALIZE_INBALANCE;

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

### Pre-processor flags

#### proba. type

The `PROBA_TYPE` flag is used to change the type used to represent probabilities and magnitude in the program. The default is `double` for a high enough precision, but other float type could be used to either get more precision or less memory/execution time cost.

#### hash map overhead

The `HASH_MAP_OVERHEAD` flag represent the overhead per element of the hashmap (of type `robin_hood::unordered_map`, default is set to `1.7` which has been determined through experiments).

### Global variables

The default value of any of those variable can be altered at compilation, by passing an uppercase flag with the same name as the desired variable.

#### tolerance

`tolerance` represents the minimum probability considered non-zero (default is `1e-30`, to compensate for numerical errors).

#### safety margin

`safety_margin` represents the target proportion of memory to keep free (default is `0.1` for 10%).

#### simple truncation

`simple_truncation` is a `bool` variable (default is `false`, but can be set to `true` by compilling with the `SIMPLE_TRUNCATION` flag). If `simple_truncation` is `true`, then truncation simply consist in selecting the n highest probability objects. Otherwise object are selected with some probabilistic aspect, with the probability of keeping an object being proportional to the probability of each object.

Probabilistic selctions cost a bit of time and of accuracy, with some gain in representation through an analog sampling process to a quantum Monte-Carlo algorithm.

#### load balancing bucket per thread

`load_balancing_bucket_per_thread` represent the number of partition par thread (or MPI node), which allows load balancing by then having a variable number of partition per thread according to each partition's size.

`load_balancing_bucket_per_thread` has a default of `8`.

### MPI global variables

#### minimum equalize size and equalize imbalance.

`mpi::min_equalize_size` represents the minimum per node average size required to automaticly call `equalize(...)` after a call to `quids::mpi::simulate(...)`.

If this first condition is met, `equalize(...)` if the maximum relative imbalance in the number of object accross the nodes is greater than `mpi::equalize_imablance`.

`mpi::min_equalize_size` is equal to `1000` by default, and `mpi::equalize_imablance` has a default of `0.2`.

### Utils global variables

#### min vector size

`utils::min_vector_size` represent the minimum size of any vector (the default is `100000`).

#### upsize policy

`utils::upsize_policy` represent the multiplier applied when upsizing a vector (the default is `1.1`). It avoid frequent upsizing by giving a small margin.

#### downsize policy

`utils::downsize_policy` reprensent the threshold multiplier to downsize a vector (the default is `0.85`). A vector won't be downsized until the requested size is smaller than this ultiplier times the capacity of the given vector.

__!! this multiplier should always be smaller than the inverse of upsize_policy to avoid upsizing-downsizing loop !!__