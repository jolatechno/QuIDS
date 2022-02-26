float unfiorm_from_hash(size_t hash) {
	static float max_size_t = (float)((size_t)0xffffffffffffffff);
	return ((float)hash) / max_size_t; 
}

float unfiorm_from_float(float x) {
	float *x_ptr = &x;
	int x_int = *(int*)x_ptr;
	x_int = x_int ^ (x_int << 16) ^ (x_int >> 16);

	static float max_int = (float)((int)0xffffffff);
	return ((float)x_int) / max_int; 
}