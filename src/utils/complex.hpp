#pragma once

template<typename T>
void inline time_equal(T& r_res, T& i_res, T r_mul, T i_mul) {
	T temp = r_res;
	r_res = temp*r_mul - i_res*i_mul;
	i_res = temp*i_mul + i_res*r_mul;
}