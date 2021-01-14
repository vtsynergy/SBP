#include "evaluate.hpp"

double evaluate::calculate_f1_score(const Graph &graph, Hungarian::Matrix &contingency_table) {
    // The number of vertex pairs = |V| choose 2 = |V|! / (2! * (|V| - 2)!) = (|V| * |V|-1) / 2
    double num_pairs = (graph.num_vertices * (graph.num_vertices - 1)) / 2;

    const int nrows = contingency_table.size();
    const int ncols = contingency_table[0].size();
    std::vector<int> rowsums(nrows, 0);
    std::vector<int> colsums(ncols, 0);
    double cell_pairs = 0;
    double cells_squared = 0;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            int value = contingency_table[row][col];
            rowsums[row] += value;
            colsums[col] += value;
            cell_pairs += (value * (value - 1));
            cells_squared += (value * value);
        }
    }
    cell_pairs /= 2;
    double row_pairs = 0;  // num_same_in_b1
    double col_pairs = 0;  // num_same_in_b2
    double rowsums_squared = 0;
    double colsums_squared = 0;
    for (int i = 0; i < nrows; ++i) {
        row_pairs += (rowsums[i] * (rowsums[i] - 1));
        rowsums_squared += (rowsums[i] * rowsums[i]);
    }
    for (int i = 0; i < ncols; ++i) {
        col_pairs += (colsums[i] * (colsums[i] - 1));
        colsums_squared += (colsums[i] * colsums[i]);
    }
    row_pairs /= 2;
    col_pairs /= 2;
    double num_agreement_diff = (graph.num_vertices * graph.num_vertices) + cells_squared;
    num_agreement_diff -= (rowsums_squared + colsums_squared);
    num_agreement_diff /= 2.0;
    double num_agreement = cell_pairs + num_agreement_diff;
    double rand_index = num_agreement / num_pairs;
    double recall = cell_pairs / row_pairs;
    double precision = cell_pairs / col_pairs;
    double f1_score = (2.0 * precision * recall) / (precision + recall);

    std::cout << "Rand index: " << rand_index << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "F1 Score: " << f1_score << std::endl;
    return f1_score;
}

double evaluate::evaluate_blockmodel(const Graph &graph, Blockmodel &blockmodel) {
    Hungarian::Matrix contingency_table = hungarian(graph, blockmodel);
    double f1_score = calculate_f1_score(graph, contingency_table);
    return f1_score;
}

Hungarian::Matrix evaluate::hungarian(const Graph &graph, Blockmodel &blockmodel) {
    // Create contingency table
    int num_true_communities = 0;
    std::unordered_map<int, int> translator;  // TODO: add some kind of if statement for whether to use this or not
    translator[-1] = -1;
    for (int community : graph.assignment) {
        if (community > -1) {
            if (translator.insert(std::unordered_map<int, int>::value_type(community, num_true_communities)).second) {
                num_true_communities++;
            };
        }
    }
    std::vector<int> true_assignment(graph.assignment);
    for (int i = 0; i < true_assignment.size(); ++i) {
        true_assignment[i] = translator[true_assignment[i]];
    }
    // int num_true_communities = true_communities.size();
    std::cout << "Blockmodel correctness evaluation" << std::endl;
    std::cout << "Number of vertices: " << graph.num_vertices << std::endl;
    std::cout << "Number of communities in true assignment: " << num_true_communities << std::endl;
    std::cout << "Number of communities in alg. assignment: " << blockmodel.getNum_blocks() << std::endl;
    std::vector<int> rows, cols;
    int nrows, ncols;
    if (num_true_communities < blockmodel.getNum_blocks()) {
        rows = true_assignment;
        cols = blockmodel.getBlock_assignment();
        nrows = num_true_communities;
        ncols = blockmodel.getNum_blocks();
    } else {
        rows = blockmodel.getBlock_assignment();
        cols = true_assignment;
        nrows = blockmodel.getNum_blocks();
        ncols = num_true_communities;
    }
    std::vector<std::vector<int>> contingency_table(nrows, std::vector<int>(ncols, 0));
    for (int i = 0; i < graph.num_vertices; ++i) {
        int row_block = rows[i];
        int col_block = cols[i];
        if (true_assignment[i] > -1) {
            std::cout << "row_block: " << row_block << " col block: " << col_block << std::endl;
            contingency_table[row_block][col_block]++;
        }
    }

    Hungarian::Result result = Hungarian::Solve(contingency_table, Hungarian::MODE_MAXIMIZE_UTIL);
    if (!result.success) {
        std::cout << "Failed to find solution :(" << std::endl;
        exit(-1);
    }

    std::vector<int> assignment(result.assignment.size(), 0);
    for (int row = 0; row < result.assignment.size(); ++row) {
        for (int col = 0; col < result.assignment[0].size(); ++col) {
            if (result.assignment[row][col] == 1) {
                assignment[row] = col;
            }
        }
    }

    std::vector<std::vector<int>> new_contingency_table(nrows, std::vector<int>(ncols, 0));
    for (int original_column = 0; original_column < ncols; ++original_column) {
        int new_column = assignment[original_column];
        for (int row = 0; row < nrows; ++row) {
            new_contingency_table[row][original_column] = contingency_table[row][new_column];
        }
    }

    // Make sure rows represent algorithm communities, columns represent true communities
    if (num_true_communities < blockmodel.getNum_blocks()) {
        Hungarian::Matrix transpose_contingency_table(ncols, std::vector<int>(nrows, 0));
        for (int row = 0; row < nrows; ++row) {
            for (int col = 0; col < ncols; ++col) {
                transpose_contingency_table[col][row] = new_contingency_table[row][col];
            }
        }
        new_contingency_table = transpose_contingency_table;
    }

    std::cout << "Contingency Table" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            std::cout << new_contingency_table[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return new_contingency_table;
}
