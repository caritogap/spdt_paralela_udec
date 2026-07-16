#include "histogram.cpp"
#include "constructhist.cpp"
#include "gainandimpurity.cpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <time.h>
#include <chrono>


using HistogramTable = std::vector<std::vector<Histogram>>;
using LevelHistograms = std::vector<HistogramTable>;

int main(int argc, char* argv[])
{
        const std::string dataset = argv[1];

        std::string filename_train;
        std::string filename_eval;

        int num_features = 16;
        int num_classes = 26;

        if(dataset=="letter"){
            filename_train= "data/letter-train-coded.csv";
            filename_eval= "data/letter-test-coded.csv";
            num_features = 16;
            num_classes = 26;
        }

        if(dataset=="isolet"){
            filename_train= "data/isolet-train.csv";
            filename_eval= "data/isolet-test.csv";
            num_features = 617;
            num_classes = 26;
        }

        if(dataset=="adult"){
            filename_train= "data/adult-data-onehot.csv";
            filename_eval= "data/adult-test-onehot.csv";
            num_features = 105;
            num_classes = 2;
        }

        if(dataset=="covtype"){
            filename_train= "data/covtype-train.csv";
            filename_eval= "data/covtype-test.csv";
            num_features = 54;
            num_classes = 2;
        }

        if(dataset=="epsilon"){
            filename_train= "data/epsilon_train_fixed.csv";
            filename_eval= "data/epsilon_test_fixed.csv";
            num_features = 2000;
            num_classes = 2;
        }




        int B=50;
        int B_tilde=50;


        std::vector<TreeNode> tree;
        TreeNode root;
        tree.push_back(root);

        std::vector<int> active_leaves = {0};
        int max_depth = 50;
        std::cout << "Starting tree construction.\n";
        auto total_histogram_time = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int depth = 0; depth < max_depth && !active_leaves.empty(); depth++) {
            if(depth%10 == 0){
                std::cout << "\nBuilding depth " << depth << "\n";
            }
            auto histograms_start_time = std::chrono::high_resolution_clock::now();
            LevelHistograms level_histograms = build_level_histograms_from_file(
                filename_train,
                tree,
                active_leaves,
                num_features,
                num_classes,
                B
            );
            auto histograms_end_time = std::chrono::high_resolution_clock::now();
            total_histogram_time += std::chrono::duration_cast<std::chrono::milliseconds>(histograms_end_time - histograms_start_time).count();

            std::vector<int> next_active_leaves;

            for (int k = 0; k < static_cast<int>(active_leaves.size()); k++) {
                int node_id = active_leaves[k];

                std::vector<Histogram> merged_histspfeat; //
        
                merged_histspfeat=merge_histograms_per_feature(level_histograms[k]);
                std::vector<std::vector<double>> uniform_splits_per_feature;
                
                for(int f=0; f<num_features;f++){
                uniform_splits_per_feature.push_back(std::move(merged_histspfeat[f].uniform(B_tilde)));
                }

                Split best_split = find_best_split(level_histograms[k], uniform_splits_per_feature, Criterion::Entropy);
                
                tree[node_id].predicted_class =
                    majority_class_from_counts(get_total_counts(level_histograms[k], num_classes));

                if (should_stop(
                    get_total_counts(level_histograms[k], num_classes),
                    best_split.gain,
                    0.005,
                    depth,
                    max_depth
                )) {
                    tree[node_id].is_leaf = true;
                    continue;
                }

                tree[node_id].feature = best_split.feature;
                tree[node_id].threshold = best_split.threshold;
                tree[node_id].is_leaf = false;


                int left_id = static_cast<int>(tree.size());
                TreeNode left_child;
                left_child.predicted_class = tree[node_id].predicted_class;
                tree.push_back(left_child);

                int right_id = static_cast<int>(tree.size());
                TreeNode right_child;
                right_child.predicted_class = tree[node_id].predicted_class;
                tree.push_back(right_child);

                tree[node_id].left_child = left_id;
                tree[node_id].right_child = right_id;


                next_active_leaves.push_back(left_id);
                next_active_leaves.push_back(right_id);

            }

            active_leaves = next_active_leaves;

        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout<< "Histogram construction time (total across all depths): " << total_histogram_time << " ms\n";
        std::cout << "Tree construction time: " << duration.count()-total_histogram_time << " ms\n";
        std::cout << "Total time (histogram construction + tree construction): " << duration.count() << " ms\n";

        double eval_accuracy = evaluate_tree_from_file(
            filename_eval,
            tree,
            num_features
        );
        double train_accuracy = evaluate_tree_from_file(
            filename_train,
            tree,
            num_features
        );


        std::cout << "\nEvaluation finished.\n";
        std::cout << "Evaluation accuracy: " << eval_accuracy << "\n";
        std::cout << "Training accuracy: " << train_accuracy << "\n";
        std::cout << "Final tree size (number of nodes): " << tree.size() << "\n";


        return 0;
}
