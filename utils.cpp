#include "utils.h"

// ---Preprocessing---

void MinMaxScaler::fit ( const Eigen::MatrixXd & data )
{
    _data_mins = data.colwise().maxCoeff();
    _data_maxs = data.colwise().minCoeff();
    _fitted = true;
}

void MinMaxScaler::transform ( Eigen::MatrixXd & data ) const
{
    if ( ! _fitted )
    {
        throw std::runtime_error("MinMaxScaler needs to be fitted before transform.");
    }
    data = (data.rowwise() - _data_mins).array().rowwise() /\
        (_data_maxs - _data_mins).array();
    data = data.array() * (_max - _min) + _min;
}

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

void ShuffleRows( Eigen::MatrixXd & matrix, const unsigned int random_seed )
{
    Eigen::PermutationMatrix<Eigen::Dynamic> permutation(matrix.rows());
    permutation.setIdentity();
    srand(random_seed);
    std::random_shuffle(permutation.indices().data(),\
                        permutation.indices().data()+permutation.indices().size());
    matrix = permutation * matrix;
}

// ---End Preprocessing---

// ---Evaluation---

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred )
{
    return (double)((yTrue - yPred).array() == 0).count() / yPred.rows();
}

double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs,
                  const double & h )
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

Eigen::VectorXi Predict( const Eigen::MatrixXd & probs )
{
    Eigen::VectorXi Predictions(probs.rows());
    for ( size_t r = 0; r < probs.rows(); ++r )
    {
        probs.row(r).maxCoeff(&Predictions(r));
    }
    return Predictions;
}

double RelativeError( const Eigen::MatrixXd & M1, const Eigen::MatrixXd & M2, const double & epsilon )
{
    return ( (M1 - M2).array().abs() /\
             (M1.array().abs() + M2.array().abs() + epsilon).maxCoeff() ).maxCoeff();
}

// ---End Evaluation---
