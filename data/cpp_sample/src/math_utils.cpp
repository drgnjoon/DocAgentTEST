#include "math_utils.h"

#include <numeric>
#include <sstream>

namespace docagent {

Stats::Stats() : count_(0), total_(0), samples_() {}

void Stats::add(int value) {
    ++count_;
    total_ += value;
    samples_.push_back(value);
}

double Stats::average() const {
    if (count_ == 0) {
        return 0.0;
    }
    return static_cast<double>(total_) / static_cast<double>(count_);
}

std::string Stats::report(const std::string& label) const {
    std::ostringstream out;
    out << label << ": " << count_ << " samples, avg=" << average();
    return out.str();
}

int add(int lhs, int rhs) {
    return lhs + rhs;
}

int sum(const std::vector<int>& values) {
    return std::accumulate(values.begin(), values.end(), 0);
}

}  // namespace docagent
