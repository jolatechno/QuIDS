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
/* variables*/
iqs::it_t buffer, state;
iqs::sy_it_t symbolic_iteration;

/* initializing the state */
state.append(object_begin, object_end);

/* applying a modifier */
iqs::simulate(state, my_modifier);
iqs::simulate(state, [](char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		/* using lambda-expressions */
	});

/* applying a rule */
iqs::rule_t *rule = new my_rule(/*...*/);
iqs::simulate(state, rule, buffer, symbolic_iteration);
```

### Modifiers

A `modifier` is a simple functions that takes a objects, and modify it in place, while keep its size unchanged.

```cpp
void my_modifier(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
	// modify the object...
}
```

### Rule

A `rule` is a simple class, implementing 2 functuions (with a third being optional).

```cpp
class my_rule : public iqs::rule_t {
public:
	my_rule() {};
	inline void get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const override;
	inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override;
	inline size_t hasher(char* parent_begin, char* parent_end) const; // optional
};
```

The first function, `get_num_child(...)`, finds the number of objects created through the unitary transform by a given objects. It also gives an upper-bound to the size of those objects. 

```cpp
inline void my_rule::get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const override {
	// do stuff...
	num_child = actual_num_child;
	max_child_size = actual_max_child_size;
}
```

The second function, `populate_child(...)`, simply "populate" (create) an object from its parent, while also modifying its magnitude according to the unitary transformation (the starting magnitude is the parent magnitude).

Note that `child_id` is a number from `0` to `num_child`, and simply identify "child" amoung its "siblings" (objects with the same "parent").

```cpp
inline char* my_rule::populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
	// modify imag and real...
	// populate the child, starting at child_begin
	return child_end;
}
```

The last function is a hasher for objects. If not specified, the whole memory buffer will simply be hashed. A hasher NEEDS to be provided if objects that are equal can be represented by different objects. The default implementation is:

```cpp
inline size_t my_rule::hasher(char* parent_begin, char* parent_end) const override { //can be overwritten
	return std::hash<std::string_view>()(std::string_view(parent_begin, std::distance(parent_begin, parent_end)));
}```