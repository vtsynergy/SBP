#include "evaluate.hpp"

double evaluate::calculate_f1_score(int num_vertices, Hungarian::Matrix &contingency_table) {
    // The number of vertex pairs = |V| choose 2 = |V|! / (2! * (|V| - 2)!) = (|V| * |V|-1) / 2
    double num_pairs = (num_vertices * (num_vertices - 1.0)) / 2.0;
    std::cout << "num_pairs = " << num_pairs << std::endl;
    const int nrows = contingency_table.size();
    const int ncols = contingency_table[0].size();
    std::vector<double> rowsums(nrows, 0);
    std::vector<double> colsums(ncols, 0);
    double cell_pairs = 0;  // num_agreement_same
    double cells_squared = 0;  // sum_table_squared
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            int value = contingency_table[row][col];
            rowsums[row] += value;
            colsums[col] += value;
            cell_pairs += (value * (value - 1));
            cells_squared += (value * value);
        }
    }
    std::cout << "sum_table_squared = " << cells_squared << std::endl;
    cell_pairs /= 2.0;
    double row_pairs = 0;  // num_same_in_b1
    double col_pairs = 0;  // num_same_in_b2
    double rowsums_squared = 0;  // sum_rowsum_squared
    double colsums_squared = 0;  // sum_colsum_squared
    std::cout << "limit<int> = " << std::numeric_limits<int>::max() << " " << std::numeric_limits<double>::max() << " " << std::numeric_limits<float>::max() << std::endl;
    for (double sum : rowsums) {
        row_pairs += (sum * (sum - 1));
        std::cout << rowsums_squared << " += (" << sum << "^2 =) " << sum * sum << " = " << rowsums_squared + (sum * sum) << std::endl;
        rowsums_squared += (sum * sum);
    }
    for (double sum : colsums) {
        col_pairs += (sum * (sum - 1));
        colsums_squared += (sum * sum);
    }
//    for (int i = 0; i < nrows; ++i) {
//        row_pairs += (rowsums[i] * (rowsums[i] - 1));
//        rowsums_squared += (rowsums[i] * rowsums[i]);
//    }
//    for (int i = 0; i < ncols; ++i) {
//        col_pairs += (colsums[i] * (colsums[i] - 1));
//        colsums_squared += (colsums[i] * colsums[i]);
//    }
    std::cout << "sum_colsum_squared = " << colsums_squared << std::endl;
    utils::print<double>(rowsums);
    std::cout << "sum_rowsum_squared = " << rowsums_squared << std::endl;
    row_pairs /= 2.0;
    col_pairs /= 2.0;
    std::cout << "num_same_in_b1 = " << row_pairs << std::endl;
    std::cout << "num_same_in_b2 = " << col_pairs << std::endl;
    double num_agreement_diff = (num_vertices * num_vertices) + cells_squared;
    num_agreement_diff -= (rowsums_squared + colsums_squared);
    num_agreement_diff /= 2.0;
    double num_agreement = cell_pairs + num_agreement_diff;
    double rand_index = num_agreement / num_pairs;
    double recall = cell_pairs / row_pairs;
    double precision = cell_pairs / col_pairs;
    double f1_score = (2.0 * precision * recall) / (precision + recall);

    std::cout << "Rand index: " << rand_index << std::endl;

    // TODO: precision & recall could be flipped. Figure out when that is so...
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "F1 Score: " << f1_score << std::endl;
    return f1_score;
}

double evaluate::evaluate_blockmodel(const Graph &graph, Blockmodel &blockmodel) {
    Hungarian::Matrix contingency_table = hungarian(graph, blockmodel);
    double f1_score = calculate_f1_score(graph.num_vertices(), contingency_table);
    return f1_score;
}

Hungarian::Matrix evaluate::hungarian(const Graph &graph, Blockmodel &blockmodel) {
    // Create contingency table
    int num_true_communities = 0;
    std::unordered_map<int, int> translator;  // TODO: add some kind of if statement for whether to use this or not
    translator[-1] = -1;
    for (int community : graph.assignment()) {
        if (community > -1) {
            if (translator.insert(std::unordered_map<int, int>::value_type(community, num_true_communities)).second) {
                num_true_communities++;
            };
        }
    }
    std::vector<int> true_assignment(graph.assignment());
    for (size_t i = 0; i < true_assignment.size(); ++i) {
        true_assignment[i] = translator[true_assignment[i]];
    }
    std::cout << "Blockmodel correctness evaluation" << std::endl;
    std::cout << "Number of vertices: " << graph.num_vertices() << std::endl;
    std::cout << "Number of communities in true assignment: " << num_true_communities << std::endl;
    std::cout << "Number of communities in alg. assignment: " << blockmodel.getNum_blocks() << std::endl;
    std::vector<int> rows, cols;
    int nrows, ncols;
    if (num_true_communities < blockmodel.getNum_blocks()) {
        rows = true_assignment;
        cols = blockmodel.block_assignment();
        nrows = num_true_communities;
        ncols = blockmodel.getNum_blocks();
    } else {
        rows = blockmodel.block_assignment();
        cols = true_assignment;
        nrows = blockmodel.getNum_blocks();
        ncols = num_true_communities;
    }
    std::vector<std::vector<int>> contingency_table(nrows, std::vector<int>(ncols, 0));
    for (int i = 0; i < graph.num_vertices(); ++i) {
        int row_block = rows[i];
        int col_block = cols[i];
        if (true_assignment[i] > -1) {
            contingency_table[row_block][col_block]++;
        }
    }

    Hungarian::Result result = Hungarian::Solve(contingency_table, Hungarian::MODE_MAXIMIZE_UTIL);
    if (!result.success) {
        std::cout << "Failed to find solution :(" << std::endl;
        exit(-1);
    }

    std::vector<int> assignment(result.assignment.size(), 0);
    for (size_t row = 0; row < result.assignment.size(); ++row) {
        for (size_t col = 0; col < result.assignment[0].size(); ++col) {
            if (result.assignment[row][col] == 1) {
                assignment[row] = (int) col;
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
        int temp = ncols;
        ncols = nrows;
        nrows = temp;
    }

    std::cout << "Contingency Table" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        if (nrows > 50) {
            if (i == 25) std::cout << "...\n";
            if (i > 25 && i < (nrows - 25)) continue;
        }
        for (int j = 0; j < ncols; ++j) {
            std::cout << new_contingency_table[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return new_contingency_table;
}
