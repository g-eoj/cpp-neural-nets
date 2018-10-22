#include "two_layer_net.h"

double TwoLayerNet::loss( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y )
{
    return (_softmax.output(_h1.output(input)).array().log() *\
            -y.array()).rowwise().sum().mean();
}

double TwoLayerNet::loss( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y, ForwardPass & fp )
{
    fp.h1 = _h1.output(input);
    fp.probs = _softmax.output(fp.h1);
    return (fp.probs.array().log() * -y.array()).rowwise().sum().mean();
}

void TwoLayerNet::backpass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y, const ForwardPass & fp, BackwardPass & bp )
{
    // softmax layer gradients
    bp.dscores = (fp.probs - y).array() / y.rows();
    bp.db2 = bp.dscores.colwise().sum();
    bp.dW2 = fp.h1.transpose() * bp.dscores;
    // hidden layer gradients
    bp.dh1 = (fp.h1.array() > 0).select(bp.dscores * _softmax.W().transpose(), 0);
    bp.db1 = bp.dh1.colwise().sum();
    bp.dW1 = input.transpose() * bp.dh1;
}

void TwoLayerNet::update( const Updates & u )
{
    _h1.update(LayerParams::WEIGHTS, u.W1);
    _h1.update(LayerParams::BIAS, u.b1);
    _softmax.update(LayerParams::WEIGHTS, u.W2);
    _softmax.update(LayerParams::BIAS, u.b2);
}

void TwoLayerNet::update( NetParams np, const Eigen::MatrixXd & u )
{
    switch ( np ) {
    case NetParams::W1:
        _h1.update(LayerParams::WEIGHTS, u);
        break;
    case NetParams::b1:
        _h1.update(LayerParams::BIAS, u);
        break;
    case NetParams::W2:
        _softmax.update(LayerParams::WEIGHTS, u);
        break;
    case NetParams::b2:
        _softmax.update(LayerParams::BIAS, u);
        break;
    }
}

double GradCheck ( NetParams np, const Eigen::MatrixXd & analytic_grad, TwoLayerNet & net, const Eigen::MatrixXd & x, const Eigen::MatrixXd & y)
{
    double h = 1e-06;
    int rc = analytic_grad.rows();
    int cc = analytic_grad.cols();
    Eigen::MatrixXd numeric_grad(rc, cc);
    Eigen::MatrixXd h_matrix = Eigen::MatrixXd::Zero(rc, cc);
    double np_plus_h;
    double np_minus_h;

    for ( int i = 0; i < rc * cc; ++i )
    {
        h_matrix(i) = h;
        net.update(np, h_matrix);
        np_plus_h = net.loss(x, y);

        h_matrix(i) = -h * 2;
        net.update(np, h_matrix);
        np_minus_h = net.loss(x, y);

        h_matrix(i) = h;
        net.update(np, h_matrix);

        h_matrix(i) = 0;

        numeric_grad(i) = (np_plus_h - np_minus_h) / (2 * h);
    }

    double relatve_error;
    relatve_error = ( (numeric_grad - analytic_grad).array().abs() /\
                      (numeric_grad.array().abs() +\
                       analytic_grad.array().abs() + h).maxCoeff() ).maxCoeff();
    return relatve_error;
}
