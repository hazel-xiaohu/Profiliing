#include"pangulu.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

int main(int ARGC ,char **ARGV){
    bool has_matrix_file = false;
    bool has_block_size = false;
    for (int i = 1; i < ARGC; ++i) {
        if (std::strcmp(ARGV[i], "-F") == 0 && i + 1 < ARGC) {
            has_matrix_file = true;
            ++i;
            continue;
        }

        if (std::strcmp(ARGV[i], "-O") == 0 && i + 1 < ARGC && std::atoi(ARGV[i + 1]) > 0) {
            has_block_size = true;
            ++i;
            continue;
        }
    }

    if (!has_matrix_file || !has_block_size) {
        std::fprintf(stderr, "Usage: %s -F <matrix_file> -O <block_size>\n", ARGV[0]);
        std::fprintf(stderr, "Example: %s -F ../test/olm100/olm100.mtx -O 32\n", ARGV[0]);
        return 1;
    }

    setenv("HWLOC_COMPONENTS", "-gl", 0);
    pangulu(ARGC ,ARGV);
    return 0;
}
