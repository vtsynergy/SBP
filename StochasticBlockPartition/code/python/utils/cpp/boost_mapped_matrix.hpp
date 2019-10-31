#include "csparse_matrix.hpp"

/****
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
class BoostMappedMatrix : public CSparseMatrix {
public:
    BoostMappedMatrix(int nrows, int ncols) : ncols(ncols), nrows(nrows) {
        this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        int shape_array[2] = {this->nrows, this->ncols};
        this->shape = py::array_t<int>(2, shape_array);
    }
    BoostMappedMatrix copy();
    py::array_t<int> getrow(int row);
    py::array_t<int> getcol(int col);
    void update_edge_counts(int current_block, int proposed_block, py::array_t<int> current_row,
        py::array_t<int> proposed_row, py::array_t<int> current_col, py::array_t<int> proposed_col);
    py::tuple nonzero();
    py::array_t<int> values();
    int sum();
    py::array_t<int> sum(int axis = 0);
    void sub(int row, int col, int val);
    void add(int row, int col, int val);
    void add(int row, py::array_t<int> cols, py::array_t<int> values);
    int operator[] (py::tuple index);
private:
    void check_row_bounds(int row);
    void check_col_bounds(int col);
    int ncols;
    int nrows;
    boost::numeric::ublas::mapped_matrix<int> matrix;
};