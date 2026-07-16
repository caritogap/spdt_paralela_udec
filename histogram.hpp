#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <vector>
#include <vector>

struct Bin {
    double p;
    double m;
};

class Histogram {
private:
    int B;
    std::vector<Bin> bins;

    void sort_bins();
    void merge_closest_bins();

public:
    explicit Histogram(int num_bins);
    void update(double x);
    Histogram merge(const Histogram& other) const;
    double sum_less_equal(double b) const;
    std::vector<double> uniform(int B_tilde) const;
    int size();
    std::vector<Bin> get_bins() const;
    void merge_equal_position_bins();
    void print() const;
    double total_mass() const;
};

#endif
