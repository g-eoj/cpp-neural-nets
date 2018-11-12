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
    unsigned int numLabels = 1 + labels.maxCoeff() - labels.minCoeff();
    Eigen::MatrixXd oh = Eigen::MatrixXd::Zero(labels.size(), numLabels);
    for ( int l = 0; l < labels.size(); ++l )
    {
        oh(l, labels(l) - 1) = 1;
    }
    return oh;
}

// Predict class integer label based on highest probability from forward pass.
Eigen::VectorXi Predict( const Eigen::MatrixXd & probs )
{
    Eigen::VectorXi Predictions(probs.rows());
    for ( int r = 0; r < probs.rows(); ++r )
    {
        probs.row(r).maxCoeff(&Predictions(r));
    }
    return Predictions;
}

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred )
{
    return (double)((yTrue - yPred).array() == 0).count() / yPred.rows();
}

// Returns relative error between numerical gradient and analytic gradient.
// This function is computationally expensive. The only reason to use it is to
// check if Layer::backward_pass() is implemented correctly.
double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd analytic_grad,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs)
{
    const double h = 1e-06;
    const int rc = analytic_grad.rows();
    const int cc = analytic_grad.cols();
    Eigen::MatrixXd numeric_grad(rc, cc);
    Eigen::MatrixXd h_matrix = Eigen::MatrixXd::Zero(rc, cc);
    double lp_plus_h;
    double lp_minus_h;

    for ( int i = 0; i < rc * cc; ++i )
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

    double relatve_error;
    relatve_error = ( (numeric_grad - analytic_grad).array().abs() /\
                      (numeric_grad.array().abs() +\
                       analytic_grad.array().abs() + h).maxCoeff() ).maxCoeff();
    return relatve_error;
}

#endif
