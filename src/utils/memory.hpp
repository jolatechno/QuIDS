struct sysinfo {
	unsigned long totalram = 0;
	unsigned long freeram = 0;
	unsigned long memavailable = 0;
	unsigned long bufferram = 0;
	unsigned long cacheram = 0;
	const unsigned long mem_unit = 1000; // 1kB
};

int sysinfo(sysinfo *info) {
	char buff[128];
	char useless[128];

	FILE *fd = fopen("/proc/meminfo", "r");

	fgets(buff, sizeof(buff), fd); 
	sscanf(buff, "%s %lu ", useless, &info->totalram);
	fgets(buff, sizeof(buff), fd); 
	sscanf(buff, "%s %lu ", useless, &info->freeram);  
	fgets(buff, sizeof(buff), fd);
	sscanf(buff, "%s %lu ", useless, &info->memavailable);  
	fgets(buff, sizeof(buff), fd);
	sscanf(buff, "%s %lu ", useless, &info->bufferram);
	fgets(buff, sizeof(buff), fd);
	sscanf(buff, "%s %lu ", useless, &info->cacheram);

	fclose(fd);

    return 0;
}

void inline get_mem_usage_and_free_mem(long long int &total_ram, long long int &free_ram) {
	struct sysinfo info;
	if (sysinfo(&info) < 0)
		throw;

    total_ram = info.totalram * info.mem_unit;
    free_ram = info.memavailable * info.mem_unit; //(info.freeram + info.bufferram + info.cacheram) * info.mem_unit;
}