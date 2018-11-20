#ifndef _UTILS
#define _UTILS

#include <Eigen/Core>
#include <fstream>
#include <vector>

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
Eigen::MatrixXd OneHot( const Eigen::VectorXi & labels )
{
    unsigned int min_label = labels.minCoeff();
    unsigned int numLabels = 1 + labels.maxCoeff() - min_label;
    Eigen::MatrixXd oh = Eigen::MatrixXd::Zero(labels.size(), numLabels);
    for ( size_t l = 0; l < labels.size(); ++l )
    {
        oh(l, labels(l) - min_label) = 1;
    }
    return oh;
}

// Predict class integer label based on highest probability from forward pass.
Eigen::VectorXi Predict( const Eigen::MatrixXd & probs )
{
    Eigen::VectorXi Predictions(probs.rows());
    for ( size_t r = 0; r < probs.rows(); ++r )
    {
        probs.row(r).maxCoeff(&Predictions(r));
    }
    return Predictions;
}

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred )
{
    return (double)((yTrue - yPred).array() == 0).count() / yPred.rows();
}

double RelativeError( const Eigen::MatrixXd & M1, const Eigen::MatrixXd & M2, const double & epsilon = 1e-06 )
{
    return ( (M1 - M2).array().abs() /\
             (M1.array().abs() + M2.array().abs() + epsilon).maxCoeff() ).maxCoeff();
}

// Returns relative error between numerical gradient and analytic gradient.
// This function is computationally expensive. The only reason to use it is to
// check if each Layer::backward_pass() is implemented correctly. Update the
// gradients in `net` with `net.probs(input, true_probs)` before passing.
double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs,
                  const double & h = 1e-06 )
{
    const LayerGradients & analytic_grads = net.gradients().at(layer_index);
    const size_t rc = ( lp == LayerParams::WEIGHTS ) ? analytic_grads.W.rows() : analytic_grads.b.size();
    const size_t cc = ( lp == LayerParams::WEIGHTS ) ? analytic_grads.W.cols() : 1;
    Eigen::MatrixXd numeric_grad(rc, cc);
    Eigen::MatrixXd h_matrix = Eigen::MatrixXd::Zero(rc, cc);
    double lp_plus_h;
    double lp_minus_h;
    for ( size_t i = 0; i < rc * cc; ++i )
    {
        h_matrix(i) = h;
        net.update(layer_index, lp, h_matrix);
        lp_plus_h = net.loss(net.probs(input), true_probs);

        h_matrix(i) = -h * 2;
        net.update(layer_index, lp, h_matrix);
        lp_minus_h = net.loss(net.probs(input), true_probs);

        h_matrix(i) = h;
        net.update(layer_index, lp, h_matrix);

        h_matrix(i) = 0;

        numeric_grad(i) = (lp_plus_h - lp_minus_h) / (2 * h);
    }
    return ( lp == LayerParams::WEIGHTS ) ? RelativeError(numeric_grad, analytic_grads.W) : RelativeError(numeric_grad, analytic_grads.b);
}

void ShuffleRows( Eigen::MatrixXd & matrix )
{
    Eigen::PermutationMatrix<Eigen::Dynamic> permutation(matrix.rows());
    permutation.setIdentity();
    std::random_shuffle(permutation.indices().data(),\
                        permutation.indices().data()+permutation.indices().size());
    matrix = permutation * matrix;
}

#endif
