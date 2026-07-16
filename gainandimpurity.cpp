#include "histogram.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

enum class Criterion {
    Gini,
    Entropy
};



double impurity(const std::vector<double>& class_counts, Criterion criterion) {
    double total = 0.0;

    for (double count : class_counts) {
        total += count;
    }

    if (total == 0.0) {
        return 0.0;
    }

    if (criterion == Criterion::Gini) {
        double sum_sq = 0.0;

        for (double count : class_counts) {
            double q = count / total;
            sum_sq += q * q;
        }

        return 1.0 - sum_sq;
    }

    if (criterion == Criterion::Entropy) {
        double entropy = 0.0;

        for (double count : class_counts) {
            if (count > 0.0) {
                double q = count / total;
                entropy -= q * std::log(q);
            }
        }

        return entropy;
    }

    throw std::runtime_error("Unknown impurity criterion.");
}

double gain(
    const std::vector<double>& parent_counts,
    const std::vector<double>& left_counts,
    const std::vector<double>& right_counts,
    Criterion criterion
) {
    double parent_total = 0.0;
    double left_total = 0.0;
    double right_total = 0.0;

    for (double count : parent_counts) parent_total += count;
    for (double count : left_counts) left_total += count;
    for (double count : right_counts) right_total += count;

    if (parent_total == 0.0 || left_total == 0.0 || right_total == 0.0) {
        return 0.0;
    }

    double parent_impurity = impurity(parent_counts, criterion);
    double left_impurity = impurity(left_counts, criterion);
    double right_impurity = impurity(right_counts, criterion);

    return parent_impurity
           - (left_total / parent_total) * left_impurity
           - (right_total / parent_total) * right_impurity;
}


struct Split {
    int feature;
    double threshold;
    double gain;
};

Split find_best_split_parallel(
    const std::vector<std::vector<Histogram>>& histograms,
    const std::vector<std::vector<double>>& splits_per_feature,
    Criterion criterion,
    int num_threads
) {
    const int num_features =
        static_cast<int>(histograms.size());

    const int num_classes =
        static_cast<int>(histograms[0].size());

    Split global_best;
    global_best.feature = -1;
    global_best.threshold = 0.0;
    global_best.gain =
        -std::numeric_limits<double>::infinity();

    #pragma omp parallel num_threads(num_threads)
    {
        Split thread_best;
        thread_best.feature = -1;
        thread_best.threshold = 0.0;
        thread_best.gain =
            -std::numeric_limits<double>::infinity();

        std::vector<double> parent_counts(num_classes);
        std::vector<double> left_counts(num_classes);
        std::vector<double> right_counts(num_classes);

        #pragma omp for schedule(dynamic)
        for (int f = 0; f < num_features; ++f) {
            const auto& thresholds = splits_per_feature[f];

            for (int c = 0; c < num_classes; ++c) {
                parent_counts[c] =
                    histograms[f][c].total_mass();
            }

            for (double threshold : thresholds) {
                for (int c = 0; c < num_classes; ++c) {
                    const double left =
                        histograms[f][c]
                            .sum_less_equal(threshold);

                    left_counts[c] = left;
                    right_counts[c] =
                        parent_counts[c] - left;
                }

                const double split_gain = gain(
                    parent_counts,
                    left_counts,
                    right_counts,
                    criterion
                );

                if (split_gain > thread_best.gain) {
                    thread_best.feature = f;
                    thread_best.threshold = threshold;
                    thread_best.gain = split_gain;
                }
            }
        }

        #pragma omp critical
        {
            if (thread_best.gain > global_best.gain) {
                global_best = thread_best;
            }
        }
    }

    return global_best;
}
Split find_best_split(
    const std::vector<std::vector<Histogram>>& histograms,
    const std::vector<std::vector<double>>& splits_per_feature,
    Criterion criterion
) {
    const int num_features = static_cast<int>(histograms.size());
    const int num_classes = static_cast<int>(histograms[0].size());

    Split global_best;
    global_best.gain = -std::numeric_limits<double>::infinity();


        Split local_best;
        local_best.gain = -std::numeric_limits<double>::infinity();

        std::vector<double> parent_counts(num_classes);
        std::vector<double> left_counts(num_classes);
        std::vector<double> right_counts(num_classes);


        for (int f = 0; f < num_features; ++f) {
            const auto& thresholds = splits_per_feature[f];

            for (int c = 0; c < num_classes; ++c) {
                parent_counts[c] = histograms[f][c].total_mass();
            }

            for (double threshold : thresholds) {
                for (int c = 0; c < num_classes; ++c) {
                    const double left =
                        histograms[f][c].sum_less_equal(threshold);

                    left_counts[c] = left;
                    right_counts[c] = parent_counts[c] - left;
                }

                const double split_gain =
                    gain(
                        parent_counts,
                        left_counts,
                        right_counts,
                        criterion
                    );

                if (split_gain > local_best.gain) {
                    local_best.feature = f;
                    local_best.threshold = threshold;
                    local_best.gain = split_gain;
                }
            }
        }


            if (local_best.gain > global_best.gain) {
                global_best = local_best;
            }

    

    return global_best;
}
