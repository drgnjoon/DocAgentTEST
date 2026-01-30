#ifndef DOCAGENT_CPP_SAMPLE_MATH_UTILS_H
#define DOCAGENT_CPP_SAMPLE_MATH_UTILS_H

#include <vector>
#include <string>

namespace docagent {

/**
 * Represents a running summary of integer values.
 */
class Stats {
public:
    Stats();

    void add(int value);

    double average() const;

    std::string report(const std::string& label) const;

private:
    int count_;
    int total_;
    std::vector<int> samples_;
};

int add(int lhs, int rhs);

int sum(const std::vector<int>& values);

}  // namespace docagent

#endif  // DOCAGENT_CPP_SAMPLE_MATH_UTILS_H
