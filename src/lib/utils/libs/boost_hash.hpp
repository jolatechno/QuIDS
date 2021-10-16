/* reverse-engineered from:
	https://github.com/boostorg/container_hash/blob/develop/include/boost/container_hash/hash.hpp
	(the source code of <boost/container_hash/hash>)

used to prevent the need to import boost on non-root machines, and might give a performance advantage by not calling std::hash on an int (which is unchanged by it anyway). */

namespace boost {
    inline void hash_combine(std::size_t& seed, size_t const value_64) {
	    static size_t magic_number = 0xc6a4a7935bd1e995;
	    static auto shift = 47;

	    seed *= magic_number;
	    seed ^= value_64 >> shift;
	    seed *= magic_number;

	    seed ^= value_64;
	    seed *= magic_number;

	    // Completely arbitrary number, to prevent 0's
	    // from hashing to 0.
	    seed += 0xe6546b64;
    }
}