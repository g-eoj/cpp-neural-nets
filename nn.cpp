#include "nn.h"

Layer::Layer( int input_size, int output_size )
{
    srand(time(NULL));
    double limit = sqrt(6. / ((double)input_size + (double)output_size)); // glorot uniform
    _W = Eigen::MatrixXd::Random(input_size, output_size).array() * limit;
    _b = Eigen::VectorXd::Zero(output_size);
}

void Layer::update( LayerParams lp, const Eigen::MatrixXd & update )
{
    switch ( lp ) {
    case LayerParams::WEIGHTS:
        _W.array() += update.array();
        break;
    case LayerParams::BIAS:
        _b += update;
        break;
    }
}

Eigen::MatrixXd Hidden::forward_pass( const Eigen::MatrixXd & input ) const
{
    return ((input * W()).rowwise() + b().transpose()).array().max(0);
}

LayerGradients Hidden::backward_pass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & output, const Eigen::MatrixXd & upstream_gradient ) const
{
    LayerGradients gradients;
    Eigen::MatrixXd dM = (output.array() > 0).select(upstream_gradient, 0); // backprop relu
    gradients.b = dM.colwise().sum();
    gradients.W = input.transpose() * dM;
    gradients.input = dM * W().transpose();
    return gradients;
}

Eigen::MatrixXd Softmax::forward_pass( const Eigen::MatrixXd & input ) const
{
    Eigen::MatrixXd M = (input * W()).rowwise() + b().transpose();
    Eigen::MatrixXd exp_scores = (M.array() - M.maxCoeff()).array().exp();
    return exp_scores.array().colwise() / exp_scores.rowwise().sum().array();
}

LayerGradients Softmax::backward_pass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & probs, const Eigen::MatrixXd & true_probs ) const
{
    LayerGradients gradients;
    // gradient of cross entropy loss function with respect to layer output
    // values before softmax activation
    Eigen::MatrixXd dscores = (probs - true_probs).array() / true_probs.rows();
    gradients.b = dscores.colwise().sum();
    gradients.W = input.transpose() * dscores;
    gradients.input = dscores * W().transpose();
    return gradients;
}

Eigen::MatrixXd NeuralNet::probs( const Eigen::MatrixXd & input ) const
{
    Eigen::MatrixXd output = input;
    for ( auto layer = _layers.begin(); layer != _layers.end(); ++layer )
    {
        output = (**layer).forward_pass(output);
    }
    return output;
}

Eigen::MatrixXd NeuralNet::probs( const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs )
{
    // forward pass
    std::vector<Eigen::MatrixXd> inputs;
    inputs.push_back(input);
    for ( auto layer = _layers.begin(); layer != _layers.end(); ++layer )
    {
        inputs.push_back((**layer).forward_pass(inputs.back()));
    }

    //backward pass
    size_t layer_index = _layers.size() - 1;
    // softmax
    _gradients.at(layer_index) = _layers.at(layer_index)->\
        backward_pass(inputs.at(layer_index), inputs.back(), true_probs);
    // hidden layers
    while ( layer_index != 0 )
    {
        layer_index -= 1;
        _gradients.at(layer_index) = _layers.at(layer_index)->\
            backward_pass(inputs.at(layer_index),\
                          inputs.at(layer_index + 1),\
                          _gradients.at(layer_index + 1).input);
    }

    return inputs.back();
}

double NeuralNet::loss( const Eigen::MatrixXd & probs, const Eigen::MatrixXd & true_probs ) const
{
    return (probs.array().log() * -true_probs.array()).rowwise().sum().mean();
}

void NeuralNet::update( const std::vector<LayerUpdate> & updates )
{
    for ( size_t layer_index = 0; layer_index < _layers.size(); ++layer_index )
    {
        _layers.at(layer_index)->update(LayerParams::WEIGHTS, updates.at(layer_index).W);
        _layers.at(layer_index)->update(LayerParams::BIAS, updates.at(layer_index).b);
    }
}

void NeuralNet::update( const size_t layer_index, LayerParams lp, const Eigen::MatrixXd & update )
{
    _layers.at(layer_index)->update(lp, update);
}
