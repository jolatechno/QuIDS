/*
This is the only utilisy that is system dependent !
*/

#ifdef __CYGWIN__ // windows systems

#include <windows.h>

size_t inline get_free_mem() {
	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);
	GlobalMemoryStatusEx (&statex);

    return statex.AvailPageFile; // free virtual memory instead of free physical memory...
}

#elif defined(__linux__) // linux systems

size_t inline get_free_mem() {
	char buff[128];
	char useless[128];
	unsigned long free_mem = 0;

	FILE *fd = fopen("/proc/meminfo", "r");

	fgets(buff, sizeof(buff), fd); 
	fgets(buff, sizeof(buff), fd); 
	sscanf(buff, "%s %lu ", useless, &free_mem); 

	return free_mem * 1000; 
}

#elif defined(__unix__) // other unix systems
	#error "UNIX system other than LINUX aren't supported for now"
#elif defined(__MACH__) // mac os systems
	#error "macos isn't supported for now !"
#else // other systems
	#error "system isn't supported"
#endif

