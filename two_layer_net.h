// For creating a two layer neural network classifier.

#ifndef _TWO_LAYER_NET
#define _TWO_LAYER_NET

#include <Eigen/Core>

#include "nn.h"

enum class NetParams { W1, b1, W2, b2 };

// Stores the outputs from each layer, which can then be used for prediction
// or backpropagation.
struct ForwardPass
{
    Eigen::MatrixXd h1;
    Eigen::MatrixXd probs;
};

// Stores the gradients calculated during the backward pass.
struct BackwardPass
{
    // softmax layer gradients
    Eigen::MatrixXd dscores;
    Eigen::VectorXd db2;
    Eigen::MatrixXd dW2;
    // hidden layer gradients
    Eigen::MatrixXd dh1;
    Eigen::VectorXd db1;
    Eigen::MatrixXd dW1;
};

// Store updates calculated from gradients, to be added to weights and biases.
struct Updates
{
    Eigen::VectorXd b2;
    Eigen::MatrixXd W2;
    Eigen::VectorXd b1;
    Eigen::MatrixXd W1;
};

// Basic neural network classifier with one hidden layer.
class TwoLayerNet
{
    Hidden & _h1;
    Softmax & _softmax;
    TwoLayerNet();
    public:
        TwoLayerNet( Hidden & h1, Softmax & softmax ) :\
            _h1(h1), _softmax(softmax) {}
        // get loss without storing outputs
        double loss( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y );
        // outputs from each layer are stored in fp
        double loss( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y, ForwardPass & fp );
        // gradients for each layer are stored in bp
        void backpass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & y, const ForwardPass & fp, BackwardPass & bp );
        // update all parameters at once
        void update( const Updates & u );
        // specify parameters to update
        void update( NetParams np, const Eigen::MatrixXd & u );
};

// Returns relative error between numerical gradient and analytic gradient.
// This function is computationally expensive. The only reason to use it is to
// check if TwoLayer::backpass() is implemented correctly.
double GradCheck ( NetParams np, const Eigen::MatrixXd & analyticGrad, TwoLayerNet & net, const Eigen::MatrixXd & x, const Eigen::MatrixXd & y);

#endif
