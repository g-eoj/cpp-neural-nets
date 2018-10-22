#include "nn.h"

Layer::Layer( int input_size, int output_size )
{
    double limit = sqrt(6. / ((double)input_size + (double)output_size)); // glorot uniform
    _W = Eigen::MatrixXd::Random(input_size, output_size).array() * limit;
    _b = Eigen::VectorXd::Zero(output_size);
}

void Layer::update( LayerParams lp, const Eigen::MatrixXd & u )
{
    switch ( lp ) {
    case LayerParams::WEIGHTS:
        _W.array() += u.array();
        break;
    case LayerParams::BIAS:
        _b += u;
        break;
    }
}

Eigen::MatrixXd Hidden::output ( const Eigen::MatrixXd & input ) const
{
    return ((input * W()).rowwise() + b().transpose()).array().max(0);
}

Eigen::MatrixXd Softmax::output ( const Eigen::MatrixXd & input ) const
{
    Eigen::MatrixXd exp_scores =\
        ((input * W()).rowwise() + b().transpose()).array().exp();
    return exp_scores.array().colwise() / exp_scores.rowwise().sum().array();
}
