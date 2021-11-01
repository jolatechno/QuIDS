#include <sys/sysinfo.h>

void inline get_mem_usage_and_free_mem(long long int &total_ram, long long int &free_ram) {
	struct sysinfo info;
	if (sysinfo(&info) < 0)
		throw;

    total_ram = info.totalram * info.mem_unit;
    free_ram = (info.freeram + info.bufferram) * info.mem_unit;
}