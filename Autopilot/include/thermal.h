#include <string>
#include <fstream>
#include <stdexcept>

enum THERMAL_DEVICE {
    A0   = 0,
    CPU  = 1,
    GPU  = 2,
    PLL  = 3,
    PMIC = 4,
    FAN  = 5
};

float thermal(THERMAL_DEVICE id) {
    std::ifstream file(std::string("/sys/class/thermal/thermal_zone") + std::to_string(id) + "/temp");
    if (!file) {
        throw std::runtime_error("Tempreature device not recognized");
    }
    float temp;
    file >> temp;
    file.close();
    return temp / 1000;
}