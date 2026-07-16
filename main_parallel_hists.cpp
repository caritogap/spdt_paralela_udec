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
        int num_threads=std::stoi(argv[2]);

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
        
        
        auto total_histogram_time = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto total_find_split_time = 0;
        auto total_evaluate_split_time = 0;
        auto total_find_majority_class_time = 0;
        auto total_other_time = 0;
        auto total_merge_hist_time = 0;


        for (int depth = 0; depth < max_depth && !active_leaves.empty(); depth++) {
            // if(depth%10 == 0){
            //     std::cout << "\nBuilding depth " << depth << "\n";
            // }
            //std::cout << "\nBuilding depth " << depth << "\n";

            

            auto histograms_start_time = std::chrono::high_resolution_clock::now();
            LevelHistograms level_histograms = build_level_histograms_from_file_parallel(
                filename_train,
                tree,
                active_leaves,
                num_features,
                num_classes,
                B,
                num_threads
            );
            auto histograms_end_time = std::chrono::high_resolution_clock::now();
            auto histograms_duration = std::chrono::duration_cast<std::chrono::microseconds>(histograms_end_time - histograms_start_time);
            total_histogram_time += histograms_duration.count();


            std::vector<int> next_active_leaves;

            for (int k = 0; k < static_cast<int>(active_leaves.size()); k++) {
                //std::cout << "\nBuilding node " << active_leaves[k] << "\n";
                int node_id = active_leaves[k];

                std::vector<Histogram> merged_histspfeat; //
                auto merge_histograms_start_time = std::chrono::high_resolution_clock::now();
                merged_histspfeat=merge_histograms_per_feature_parallel(level_histograms[k], num_features,B);
                auto merge_histograms_end_time = std::chrono::high_resolution_clock::now();
                auto merge_histograms_duration = std::chrono::duration_cast<std::chrono::microseconds>(merge_histograms_end_time - merge_histograms_start_time);
                total_merge_hist_time += merge_histograms_duration.count();


                //std::cout << "Merged histograms for node " << node_id << ":\n";
                
                auto uniform_splits_start_time = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> uniform_splits_per_feature(num_features);
                #pragma omp parallel for num_threads(std::min(omp_get_max_threads(), num_features)) schedule(static)
                for (int f = 0; f < num_features; f++) {
                    uniform_splits_per_feature[f] = merged_histspfeat[f].uniform(B_tilde);
                }
                auto uniform_splits_end_time = std::chrono::high_resolution_clock::now();
                auto uniform_splits_duration = std::chrono::duration_cast<std::chrono::microseconds>(uniform_splits_end_time - uniform_splits_start_time);
                total_find_split_time += uniform_splits_duration.count();


                auto find_split_start_time = std::chrono::high_resolution_clock::now();
                Split best_split = find_best_split_parallel(level_histograms[k], uniform_splits_per_feature, Criterion::Entropy,std::min(omp_get_max_threads(), num_features));
                auto find_split_end_time = std::chrono::high_resolution_clock::now();
                auto find_split_duration = std::chrono::duration_cast<std::chrono::microseconds>(find_split_end_time - find_split_start_time);
                total_evaluate_split_time += find_split_duration.count();


                auto find_majority_class_start_time = std::chrono::high_resolution_clock::now();
                tree[node_id].predicted_class =
                    majority_class_from_counts(get_total_counts(level_histograms[k], num_classes));
                auto find_majority_class_end_time = std::chrono::high_resolution_clock::now();
                auto find_majority_class_duration = std::chrono::duration_cast<std::chrono::microseconds>(find_majority_class_end_time - find_majority_class_start_time);
                total_find_majority_class_time += find_majority_class_duration.count();
                    //std::cout << "Node " << node_id << " predicted class: " << tree[node_id].predicted_class << "\n";
                auto total_mass_start_time = std::chrono::high_resolution_clock::now();
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

                // These children will be processed in the next depth
                next_active_leaves.push_back(left_id);
                next_active_leaves.push_back(right_id);
                auto total_mass_end_time = std::chrono::high_resolution_clock::now();
                auto total_mass_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_mass_end_time - total_mass_start_time);
                total_other_time += total_mass_duration.count(); //Esto lo puse para ver que tiempo me estaba faltando. Sé que hay algunas cosas que no estoy midiendo pero estas las consideré al restar del tiempo de construcción del árbol todo el resto y así determiné la categoría "Other"

            }

            active_leaves = next_active_leaves;

        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout<< "Histogram construction time (total across all depths): " << total_histogram_time << " ms\n";

        std::cout << "Tree construction time: " << duration.count()-total_histogram_time << " ms\n";
        std::cout << "Merge histograms time: " << total_merge_hist_time << " ms\n";
        std::cout << "Find split candidates time: " << total_find_split_time << " ms\n";
        std::cout << "Evaluate split candidates time: " << total_evaluate_split_time << " ms\n";
        std::cout << "Find majority class time: " << total_find_majority_class_time << " ms\n";
        std::cout << "Other time: " << total_other_time << " ms\n";
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
