#ifndef _UTILS
#define _UTILS

#include <Eigen/Core>
#include <exception>
#include <fstream>
#include <vector>

#include "nn.h"

// ---Preprocessing---

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

void ShuffleRows( Eigen::MatrixXd & matrix );

// ---End Preprocessing---

// ---Evaluation---

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred );

// Returns relative error between numerical gradient and analytic gradient.
// This function is computationally expensive. The only reason to use it is to
// check if each Layer::backward_pass() is implemented correctly. Update the
// gradients in `net` with `net.probs(input, true_probs)` before passing.
double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs,
                  const double & h = 1e-06 );

// Predict class integer label based on highest probability from forward pass.
Eigen::VectorXi Predict( const Eigen::MatrixXd & probs );

double RelativeError( const Eigen::MatrixXd & M1, const Eigen::MatrixXd & M2, const double & epsilon = 1e-06 );

// ---End Evaluation---

#endif
