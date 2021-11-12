/*
This is the only utilisy that is system dependent !
*/

#ifdef __CYGWIN__ // windows systems

#include <windows.h>

void inline get_free_mem(size_t &free_ram) {
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);
	GlobalMemoryStatusEx (&statex);

    free_ram = statex.AvailPageFile; // free virtual memory instead of free physical memory...
}

#elif defined(__linux__) // linux systems

void inline get_free_mem(size_t &free_ram) {
	char buff[128];
	char useless[128];
	unsigned long free_mem = 0;

	FILE *fd = fopen("/proc/meminfo", "r");

	fgets(buff, sizeof(buff), fd); 
	fgets(buff, sizeof(buff), fd); 
	sscanf(buff, "%s %lu ", useless, &free_mem); 

	free_ram = free_mem * 1000; 
}

#elif defined(__unix__) // other unix systems
	#error "UNIX system other than LINUX aren't supported for now"
#elif defined(__MACH__) // mac os systems
	#error "macos isn't supported for now !"
#else // other systems
	#error "system isn't supported"
#endif

