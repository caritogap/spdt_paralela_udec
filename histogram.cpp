#include "histogram.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>


    Histogram::Histogram(int num_bins) : B(num_bins) {
        if (B <= 0) {
            throw std::invalid_argument("Number of bins must be positive.");
        }
    }

    int Histogram::size(){
        return this->bins.size();
    }


     void Histogram::sort_bins() {
        std::sort(this->bins.begin(), this->bins.end(),
                  [](const Bin& a, const Bin& b) {
                      return a.p < b.p;
                  });
    }

    void Histogram::merge_closest_bins() {


        int best_idx = 0;
        double best_dist = bins[1].p - bins[0].p;

        for (int i = 1; i < static_cast<int>(bins.size()) - 1; i++) {
            double dist = bins[i + 1].p - bins[i].p;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = i;
                }
        }

            Bin a = bins[best_idx];
            Bin b = bins[best_idx + 1];

            double new_m = a.m + b.m;
            double new_p = (a.p * a.m + b.p * b.m) / new_m;

            bins[best_idx] = {new_p, new_m};
            bins.erase(bins.begin() + best_idx + 1);
    }


    
    

    void Histogram::update(double x) {
        for (auto& bin : bins) {
            if (std::abs(bin.p - x) < 1e-6) { // Assuming a small threshold for equality
                bin.m += 1.0; // Increment mass/count
                return;
            }
        }

        // If not found, create a new bin
        bins.push_back({x, 1.0});

        
        
        if (static_cast<int>(bins.size()) > B) {
            sort_bins();
            merge_closest_bins();
        }

    }

    std::vector<Bin> Histogram::get_bins() const {
        return bins;
    }

    void Histogram::print() const {
        for (const auto& bin : bins) {
            std::cout << "(" << bin.p << ", " << bin.m << ") ";
        }
        std::cout << "\n";
    }

    void Histogram::merge_equal_position_bins() {
        if (this->bins.empty()) {
            return;
        }

        this->sort_bins();

        std::vector<Bin> merged;
        merged.push_back(this->bins[0]);

        const double eps = 1e-12;

        for (int i = 1; i < static_cast<int>(this->bins.size()); i++) {
            Bin& last = merged.back();

            if (std::abs(this->bins[i].p - last.p) < eps) {
                last.m += this->bins[i].m;
            } else {
                merged.push_back(this->bins[i]);
            }
        }

        this->bins = std::move(merged);
    }

    Histogram Histogram::merge(const Histogram& other) const {
        Histogram result(this->B);

        // 1. Copy bins from the first histogram
        result.bins = this->bins;

        // 2. Add bins from the second histogram
        for (const auto& bin : other.bins) {
            result.bins.push_back(bin);
        }

        // 3. Sort all bins by representative position p
        result.sort_bins();
        result.merge_equal_position_bins();
        // 4. Repeatedly merge closest bins until we have at most B bins
        while (static_cast<int>(result.bins.size()) > result.B) {
            result.merge_closest_bins();
        }

        return result;
    }

    double Histogram::sum_less_equal(double b) const {
        if (bins.empty()) {
            return 0.0;
        }

        // Work on a sorted copy so the original histogram is not modified.
        std::vector<Bin> sorted_bins = bins;

        std::sort(sorted_bins.begin(), sorted_bins.end(),
                [](const Bin& a, const Bin& b) {
                    return a.p < b.p;
                });

        int n = static_cast<int>(sorted_bins.size());


        if (b >= sorted_bins.back().p) {
            double total = 0.0;
            for (const auto& bin : sorted_bins) {
                 total += bin.m;
            }
            return total;
            
        }

        if (b < sorted_bins.front().p) {
            return 0.0;
        }


        int i = -1;

        for (int k = 0; k < n - 1; k++) {
            if (sorted_bins[k].p <= b && b < sorted_bins[k + 1].p) {
                i = k;
                break;
            }
        }

        if (i == -1) {
            throw std::runtime_error("No valid interval found in sum_less_equal().");
        }

        double pi = sorted_bins[i].p;
        double mi = sorted_bins[i].m;

        double pi_next = sorted_bins[i + 1].p;
        double mi_next = sorted_bins[i + 1].m;

        // Linear interpolation of the bin mass at position b
        double mb = mi + ((mi_next - mi) / (pi_next - pi)) * (b - pi);

        double s = ((mi + mb) / 2.0) * ((b - pi) / (pi_next - pi));

        for (int j = 0; j < i; j++) {
            s += sorted_bins[j].m;
        }

        s += mi / 2.0;

        return s;
    }

    std::vector<double> Histogram::uniform(int B_tilde) const {
    if (B_tilde <= 1) {
        throw std::invalid_argument("B_tilde must be greater than 1.");
    }

    std::vector<double> result;


    // Work on a sorted copy.
    std::vector<Bin> sorted_bins = bins;

    std::sort(sorted_bins.begin(), sorted_bins.end(),
              [](const Bin& a, const Bin& b) {
                  return a.p < b.p;
              });

    int n = static_cast<int>(sorted_bins.size());


    // Total number of represented points.
    double total = 0.0;
    for (const auto& bin : sorted_bins) {
        total += bin.m;
    }

    if (total == 0.0) {
        return result;
    }

    if (sorted_bins.size() < 2) {
        return result;
    }

    for (int j = 1; j < B_tilde; j++) {
        double target = (static_cast<double>(j) / B_tilde) * total;

        if (target <= 0.0) {
            result.push_back(sorted_bins.front().p);
            continue;
        }

        if (target >= total) {
            result.push_back(sorted_bins.back().p);
            continue;
        }

        // Find interval [p_i, p_{i+1}] containing the target cumulative count.
        int i = 0;
        for (; i < n - 1; i++) {
            double left_sum = sum_less_equal(sorted_bins[i].p);
            double right_sum = left_sum +(sorted_bins[i].m + sorted_bins[i + 1].m) / 2.0;

            if (left_sum <= target && target <= right_sum) {
                break;
            }
        }
        if(i == n - 1){
            result.push_back(sorted_bins.back().p);
            continue;
        }

        double pi = sorted_bins[i].p;
        double mi = sorted_bins[i].m;

        double pi_next = sorted_bins[i + 1].p;
        double mi_next = sorted_bins[i + 1].m;

        double left_sum = sum_less_equal(pi);
        double d = target - left_sum;

        double diff_m = mi_next - mi;


        double z;

        double a = diff_m;
        if (std::abs(a) < 1e-12) {
            // Linear case: mi and mi_next are approximately equal.
            if (std::abs(mi) < 1e-12) {
                z = 0.0;
            } else {
                z = d / mi;
            }
            double u = pi + (pi_next - pi) * z;

            result.push_back(u);
        }   
        else {
            double b = 2.0 * mi;
            double c = -2.0 * d;

            double discriminant = b * b - 4.0 * a * c;

            if (discriminant < 0.0) {
                throw std::runtime_error("Discriminant is negative.");
            }

            z = (-b + std::sqrt(discriminant)) / (2.0 * a);

            if (z < 0.0) z = 0.0;
            if (z > 1.0) z = 1.0;

            double u = pi + (pi_next - pi) * z;

            result.push_back(u);
        }
        
    }

    return result;
}
double Histogram::total_mass() const {
    double total = 0.0;
    for (const auto& bin : this->get_bins()) {
        total += bin.m;
    }
    return total;
}



