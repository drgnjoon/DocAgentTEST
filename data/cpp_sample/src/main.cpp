#include "math_utils.h"

#include <iostream>

int main() {
    docagent::Stats stats;
    stats.add(3);
    stats.add(7);

    std::cout << stats.report("demo") << "\n";
    std::cout << "sum=" << docagent::sum({1, 2, 3}) << "\n";
    std::cout << "add=" << docagent::add(4, 5) << "\n";
    return 0;
}
