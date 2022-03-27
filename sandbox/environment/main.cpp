#include <iostream>
#include "environment.h"

int main() {

    lumina::Environment env;

    env.add(lumina::dim(0, 0, 0));
    env.add(lumina::dim(5, 2, -7));

    env.export_stl();

    return 0;
}