#include "histogram.hpp"
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <charconv>
#include <stdexcept>


using HistogramTable = std::vector<std::vector<Histogram>>;
HistogramTable initialize_histograms(
    int num_features,
    int num_classes,
    int B
) {
    HistogramTable histograms;

    for (int feature = 0; feature < num_features; feature++) {
        std::vector<Histogram> feature_hists;

        for (int c = 0; c < num_classes; c++) {
            feature_hists.emplace_back(B);
        }

        histograms.push_back(std::move(feature_hists));
    }

    return histograms;
}



void update_histograms_with_sample(
    HistogramTable& histograms,
    const std::vector<double>& features,
    int label
) {
    int num_features = static_cast<int>(features.size());

    
    for (int feature = 0; feature < num_features; feature++) {
        histograms[feature][label].update(features[feature]);
    }
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    return tokens;
}



void stream_construct_histograms_from_file(
    const std::string& filename,
    HistogramTable& histograms
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> tokens = split_csv_line(line);

        

        char label_char = tokens[0][0];
        int label = std::stoi(tokens[0]);
        int num_features=tokens.size()-1;
        std::vector<double> features(num_features);

        for (int i = 0; i < num_features; ++i) {
                const std::string& token = tokens[i + 1];

                const char* begin = token.data();
                const char* end = begin + token.size();

                auto [ptr, ec] = std::from_chars(begin, end, features[i]);

                if (ec != std::errc{} || ptr != end) {
                    throw std::runtime_error("Invalid numeric value: " + token);
                }
    }

        update_histograms_with_sample(histograms, features, label);
        line_number++;
    }

    std::cout << "Training samples processed: " << line_number << "\n";
}

std::vector<Histogram> merge_histograms_per_feature(const HistogramTable& histograms) {
    std::vector<Histogram> merged_histograms;

    for (const auto& feature_hists : histograms) {
        if (feature_hists.empty()) {
            continue;
        }

        Histogram merged = feature_hists[0];

        for (size_t c = 1; c < feature_hists.size(); c++) {
            merged = merged.merge(feature_hists[c]);
        }

        merged_histograms.push_back(std::move(merged));
    }

    return merged_histograms;
}

std::vector<Histogram> merge_histograms_per_feature_parallel(const HistogramTable& histograms,int num_features,int B) {
    std::vector<Histogram> merged_histograms(
    num_features,
    Histogram(B)
    );
    #pragma omp parallel for num_threads(std::min(omp_get_max_threads(),num_features))
    for (int i = 0; i < num_features; i++) {
        const auto& feature_hists = histograms[i];
        if (feature_hists.empty()) {
            continue;
        }

        Histogram merged = feature_hists[0];

        for (size_t c = 1; c < feature_hists.size(); c++) {
            merged = merged.merge(feature_hists[c]);
        }

        merged_histograms[i] = std::move(merged);
    }

    return merged_histograms;
}

using LevelHistograms = std::vector<HistogramTable>;


LevelHistograms initialize_level_histograms(
    int num_active_leaves,
    int num_features,
    int num_classes,
    int B
) {
    LevelHistograms level_histograms;

    for (int k = 0; k < num_active_leaves; k++) {
        level_histograms.push_back(
            initialize_histograms(num_features, num_classes, B)
        );
    }

    return level_histograms;
}

struct TreeNode {
    bool is_leaf = false;
    int predicted_class = -1;
    int feature = -1;
    double threshold = 0.0;
    int left_child = -1;
    int right_child = -1;
};

int route_to_leaf(
    const std::vector<TreeNode>& tree,
    const std::vector<double>& features
) {
    int node_id = 0;

    while (tree[node_id].left_child != -1 &&
           tree[node_id].right_child != -1) {

        const TreeNode& node = tree[node_id];

        if (features[node.feature] <= node.threshold) {
            node_id = node.left_child;
        } else {
            node_id = node.right_child;
        }
    }

    return node_id;
}

LevelHistograms build_level_histograms_from_file(
    const std::string& filename,
    const std::vector<TreeNode>& tree,
    const std::vector<int>& active_leaves,
    int num_features,
    int num_classes,
    int B
) {
    LevelHistograms level_histograms = initialize_level_histograms(
        static_cast<int>(active_leaves.size()),
        num_features,
        num_classes,
        B
    );


    std::unordered_map<int, int> active_leaf_to_index;

    for (int k = 0; k < static_cast<int>(active_leaves.size()); k++) {
        active_leaf_to_index[active_leaves[k]] = k;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    int line_number = 0;
    int used_samples = 0;

    while (std::getline(file, line)) {
        line_number++;

        if (line.empty()) {
            continue;
        }

        std::vector<std::string> tokens = split_csv_line(line);

        char label_char = tokens[0][0];
        int label = std::stoi(tokens[0]);

        std::vector<double> features(num_features);

        for (int i = 0; i < num_features; ++i) {
                const std::string& token = tokens[i + 1];

                const char* begin = token.data();
                const char* end = begin + token.size();

                auto [ptr, ec] = std::from_chars(begin, end, features[i]);

                if (ec != std::errc{} || ptr != end) {
                    throw std::runtime_error("Invalid numeric value: " + token);
                }
        }


        int leaf_id = route_to_leaf(tree, features);
        auto it = active_leaf_to_index.find(leaf_id);
        if (it == active_leaf_to_index.end()) {
            continue;
        }

        int active_index = it->second;
        for (int feature = 0; feature < num_features; feature++) {
            level_histograms[active_index][feature][label].update(features[feature]);  
        }

        used_samples++;
    }

    return level_histograms;
}

bool should_stop(
    std::vector<double> total_counts,
    double gain,
    double min_gain,
    int depth,
    int max_depth
) {
    
    int total_samples = 0;
    for(int i = 0; i < total_counts.size(); i++)
    {
        total_samples += total_counts[i];
    }
    if (total_samples == 0.0) {
        //std::cout << "Stopping because no samples reach this node.\n";
        return true;
    }

    if (depth >= max_depth) {
        //std::cout << "Stopping because max depth reached.\n";
        return true;
    }


    if (gain < min_gain) {
        //std::cout << "Stopping because gain " << gain << " is less than min_gain " << min_gain << ".\n";
        return true;
    }

    return false;
}
std::vector<double> get_total_counts(
    const HistogramTable& histograms,
    int num_classes
) {
    std::vector<double> counts(num_classes, 0.0);

    int feature = 0;

    for (int c = 0; c < num_classes; c++) {
        for (const auto& bin : histograms[feature][c].get_bins()) {
            counts[c] += bin.m;
        }
    }

    return counts;
}

int majority_class_from_counts(const std::vector<double>& counts) {
    int best_class = 0;
    double best_count = counts[0];

    for (int c = 1; c < static_cast<int>(counts.size()); c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_class = c;
        }
    }

    return best_class;
}

int predict_one(
    const std::vector<TreeNode>& tree,
    const std::vector<double>& features
) {
    int node_id = 0;

    while (true) {
        const TreeNode& node = tree[node_id];

        // Stop if this is a leaf or if it has no children
        if (node.is_leaf || node.left_child == -1 || node.right_child == -1) {
            return node.predicted_class;
        }

        if (features[node.feature] <= node.threshold) {
            node_id = node.left_child;
        } else {
            node_id = node.right_child;
        }
    }
}

double evaluate_tree_from_file(
    const std::string& filename,
    const std::vector<TreeNode>& tree,
    int num_features
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    int total = 0;
    int correct = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (static_cast<int>(tokens.size()) != num_features + 1) {
            
            throw std::runtime_error(
                "Invalid number of columns while evaluating. Columns: " +
                std::to_string(tokens.size()) + " " + std::to_string(total)
            );
            
        }

        int true_label = std::stoi(tokens[0]);

        std::vector<double> features(num_features);

        for (int i = 0; i < num_features; ++i) {
                const std::string& token = tokens[i + 1];

                const char* begin = token.data();
                const char* end = begin + token.size();

                auto [ptr, ec] = std::from_chars(begin, end, features[i]);

                if (ec != std::errc{} || ptr != end) {
                    throw std::runtime_error("Invalid numeric value: " + token);
                }
        }

        int predicted_label = predict_one(tree, features);

        if (predicted_label == true_label) {
            correct++;
        }

        total++;
    }

    if (total == 0) {
        return 0.0;
    }

    return static_cast<double>(correct) / static_cast<double>(total);
}
LevelHistograms build_level_histograms_from_file_parallel(
    const std::string& filename,
    const std::vector<TreeNode>& tree,
    const std::vector<int>& active_leaves,
    int num_features,
    int num_classes,
    int B,
    int num_threads
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    file.close();

    int num_active = static_cast<int>(active_leaves.size());

    std::unordered_map<int, int> active_leaf_to_index;
    for (int i = 0; i < num_active; i++) {
        active_leaf_to_index[active_leaves[i]] = i;
    }

    std::vector<LevelHistograms> local_histograms(num_threads);

    for (int t = 0; t < num_threads; t++) {
        local_histograms[t].resize(num_active);

        for (int k = 0; k < num_active; k++) {
            local_histograms[t][k] =
                initialize_histograms(num_features, num_classes, B);
        }
    }
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int line_id = 0; line_id < static_cast<int>(lines.size()); line_id++) {
            std::vector<std::string> tokens = split_csv_line(lines[line_id]);
            if (static_cast<int>(tokens.size()) != num_features + 1) {
                continue;
            }

            int label = std::stoi(tokens[0]);

            if (label < 0 || label >= num_classes) {
                continue;
            }

            std::vector<double> features(num_features);
            for (int i = 0; i < num_features; ++i) {
                const std::string& token = tokens[i + 1];

                const char* begin = token.data();
                const char* end = begin + token.size();

                auto [ptr, ec] = std::from_chars(begin, end, features[i]);

                if (ec != std::errc{} || ptr != end) {
                    std::cout<<line_id<<std::endl;
                    throw std::runtime_error("Invalid numeric value: " + token);
                }
            }

            int leaf_id = route_to_leaf(tree, features);

            auto it = active_leaf_to_index.find(leaf_id);

            if (it == active_leaf_to_index.end()) {
                continue;
            }

            int active_index = it->second;

            for (int feature = 0; feature < num_features; feature++) {
                local_histograms[thread_id][active_index][feature][label].update(features[feature]);
            }
            
        }
    }


    LevelHistograms final_histograms;
    final_histograms.resize(num_active);

    for (int k = 0; k < num_active; k++) {
        final_histograms[k] =
            initialize_histograms(num_features, num_classes, B);
    }
    auto start_merge = std::chrono::high_resolution_clock::now();


    std::vector<LevelHistograms> current =std::move(local_histograms);

    while (current.size() > 1) {
        const std::size_t num_pairs = current.size() / 2;
        const bool has_unpaired = current.size() % 2 != 0;

        std::vector<LevelHistograms> next(num_pairs + (has_unpaired ? 1 : 0));

        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int pair = 0; pair < static_cast<int>(num_pairs); ++pair) {
            const std::size_t left  = 2 * pair;
            const std::size_t right = left + 1;
            next[pair] = std::move(current[left]);
  
             for (int k = 0; k < num_active; k++) {
                 for (int f = 0; f < num_features; f++) {
                    for (int c = 0; c < num_classes; c++) {
                         next[pair][k][f][c] =next[pair][k][f][c].merge(
                                 current[right][k][f][c]
                             );
                     }
               }
            }   
        }

        if (has_unpaired) {
            next.back() = std::move(current.back());
        }
        current = std::move(next);
    }
    final_histograms =std::move(current.front());

    auto end_merge= std::chrono::high_resolution_clock::now();
    auto merge_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_merge - start_merge);
    //std::cout << "Merge time: " << merge_duration.count() << " us\n";

    return final_histograms;
}

LevelHistograms build_level_histograms_from_file_parallel_vertical(
    const std::string& filename,
    const std::vector<TreeNode>& tree,
    const std::vector<int>& active_leaves,
    int num_features,
    int num_classes,
    int B,
    int num_threads
) 
{
    struct RoutedSample
    {
        int active_index;
        int label;
        std::vector<double> features;
    };
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }



    int num_active = static_cast<int>(active_leaves.size());

    std::unordered_map<int, int> active_leaf_to_index;
    for (int i = 0; i < num_active; i++) {
        active_leaf_to_index[active_leaves[i]] = i;
    }


    std::vector<RoutedSample> routed_samples;

    std::string line;
    int line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;

        if (line.empty()) {
            continue;
        }

        std::vector<std::string> tokens = split_csv_line(line);

        if (static_cast<int>(tokens.size()) != num_features + 1) {
            continue;
        }

        const int label = std::stoi(tokens[0]);

        if (label < 0 || label >= num_classes) {
            continue;
        }

        std::vector<double> features(num_features);
            auto start2 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_features; ++i) {
                const std::string& token = tokens[i + 1];

                const char* begin = token.data();
                const char* end = begin + token.size();

                auto [ptr, ec] = std::from_chars(begin, end, features[i]);

                if (ec != std::errc{} || ptr != end) {
                    throw std::runtime_error("Invalid numeric value: " + token);
                }
            }

        const int leaf_id = route_to_leaf(tree, features);
        const auto it = active_leaf_to_index.find(leaf_id);

        if (it == active_leaf_to_index.end()) {
            continue;
        }

        routed_samples.push_back({
            it->second,
            label,
            std::move(features)
        });
    }
    LevelHistograms level_histograms(num_active);

    for (int active_index = 0;
         active_index < num_active;
         ++active_index) {

        level_histograms[active_index] =
            initialize_histograms(
                num_features,
                num_classes,
                B
            );
    }
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int feature = 0; feature < num_features; ++feature) {
        for (const RoutedSample& sample : routed_samples) {
            level_histograms[sample.active_index]
                            [feature]
                            [sample.label]
                                .update(sample.features[feature]);
        }
    }

    return level_histograms;
}