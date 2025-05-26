#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"

int main(int argc, char** argv) {
    for(int i = 0; i < argc; i++) {
        fastObjMesh* obj = fast_obj_read(argv[i]);

        printf("")
    }
}