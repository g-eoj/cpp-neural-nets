#ifndef _UTILS
#define _UTILS

#include <Eigen/Core>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "nn.h"

// ---Preprocessing---

// Separate corresponding features and labels into batches.
class Batcher
{
    Eigen::MatrixXd & _X;
    Eigen::MatrixXd & _y;
    const unsigned int _batch_size;
    unsigned int _batch_begin;
    bool _shuffle;
    Batcher();
public:
    Batcher( const unsigned int batch_size, Eigen::MatrixXd & X, Eigen::MatrixXd & y, bool shuffle=true );
    // Store a batch. If batch size is larger than the remaining data,
    // the batch will only consist of the remaining data.
    void batch( Eigen::MatrixXd & X_batch, Eigen::MatrixXd & y_batch );
    const unsigned int batch_size() const { return _batch_size; }
};

// Scale feature data.
class MinMaxScaler
{
    bool _fitted;
    const double _min;
    const double _max;
    Eigen::RowVectorXd _data_mins;
    Eigen::RowVectorXd _data_maxs;
public:
    // Fitted feature data is scaled to the range [min, max].
    MinMaxScaler ( const double min=0, const double max=1 ) : _min(min), _max(max) {};
    // Compute minimum and maximum feature values to use for scaling.
    void fit ( const Eigen::MatrixXd & data );
    // Scale feature data according to fitted features and range.
    void transform ( Eigen::MatrixXd & data ) const;
};

// Load a csv into an Eigen matrix.
// Credit to user357269 on stackoverflow.
template<typename M>
M LoadCSV( const std::string & path )
{
    std::ifstream csv;
    csv.open(path);
    std::string line;
    std::vector<double> values;
    unsigned int rows = 0;
    while ( std::getline(csv, line) )
    {
        std::stringstream lineStream(line);
        std::string cell;
        while ( std::getline(lineStream, cell, ',') )
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

// One hot encode a vector of integers representing class labels.
Eigen::MatrixXd OneHot( const Eigen::VectorXi & labels );

void ShuffleRows( Eigen::MatrixXd & matrix, const unsigned int random_seed=1 );

// Split features and corresponding labels
// with optional shuffle (not stratified) before split.
void TrainTestSplit( Eigen::MatrixXd X, Eigen::MatrixXd y,\
                     Eigen::MatrixXd & X_train, Eigen::MatrixXd & X_test,\
                     Eigen::MatrixXd & y_train, Eigen::MatrixXd & y_test,\
                     float test_prop, bool shuffle=true, const unsigned int random_seed=1 );

// ---End Preprocessing---

// ---Evaluation---

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred );

// Returns relative error between numerical gradient and analytic gradient.
// This function is computationally expensive. The only reason to use it is to
// check if each Layer::backward_pass() is implemented correctly. Update the
// gradients in `net` with `net.gradients(input, true_probs)` before passing.
double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs,
                  const size_t random_seed = 1, const double & h = 1e-06 );

// Predict class integer label based on highest probability from forward pass.
Eigen::VectorXi Predict( const Eigen::MatrixXd & probs );

void PrintTrainingMetrics( const NeuralNet & net, const size_t & iteration,
                           const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & X_val,
                           const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & y_val );

// Generate random feature and label data.
// Feature matrix `X` is filled with random numbers from [0, 1].
// Label vector `y` is filled with random integers from [0, `num_classes`).
void RandomData( size_t num_examples, size_t num_features, size_t num_classes,\
                 Eigen::MatrixXd & X, Eigen::VectorXi & y );

double RelativeError( const Eigen::MatrixXd & M1, const Eigen::MatrixXd & M2, const double & epsilon = 1e-06 );

// ---End Evaluation---

#endif
