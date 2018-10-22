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

double Accuracy ( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred )
{
    return (double)((yTrue - yPred).array() == 0).count() / yPred.rows();
}

#endif
