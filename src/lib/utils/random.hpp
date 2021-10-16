double unfiorm_from_hash(size_t hash) {
	static double max_size_t = (double)((size_t)0xffffffffffffffff);
	return ((double)hash) / max_size_t; 
}