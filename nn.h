// Classes for creating layers and a neural network.

#ifndef _NN
#define _NN

#include <ctime>
#include <Eigen/Core>
#include <iostream>
#include <random>
#include <vector>

enum class LayerParams { WEIGHTS, BIAS };

struct LayerGradients
{
    Eigen::MatrixXd W;
    Eigen::VectorXd b;
    Eigen::MatrixXd input;
};

struct LayerUpdate
{
    Eigen::MatrixXd W;
    Eigen::VectorXd b;
};

// Base class for dense layers.
// Weights are initialized with Glorot uniform initializer.
// Bias is initialized to zeros.
class Layer
{
    Eigen::MatrixXd _W;
    Eigen::VectorXd _b;
    Layer();
    protected:
        Layer( int input_size, int output_size );
    public:
        virtual ~Layer() {}
        const Eigen::MatrixXd & W() const { return _W; }
        const Eigen::VectorXd & b() const { return _b; }
        // Return layer output for given input.
        virtual Eigen::MatrixXd forward_pass( const Eigen::MatrixXd &) const = 0;
        // Return gradients with respect to layer weights, bias, and input.
        virtual LayerGradients backward_pass( const Eigen::MatrixXd & , const Eigen::MatrixXd &, const Eigen::MatrixXd & ) const = 0;
        // Add update to WEIGHTS or BIAS.
        void update( LayerParams lp, const Eigen::MatrixXd & update );
};

// Dense layer with relu activation.
class Hidden : public Layer
{
    Hidden();
    public:
        Hidden( int input_size, int output_size ) :\
            Layer(input_size, output_size) {}
        // Return layer output for given input.
        Eigen::MatrixXd forward_pass( const Eigen::MatrixXd & input ) const;
        // Return gradients with respect to layer weights, bias, and input.
        LayerGradients backward_pass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & output, const Eigen::MatrixXd & upstream_gradient ) const;
};

// Dense layer with softmax activation.
class Softmax : public Layer
{
    Softmax();
    public:
        Softmax( int input_size, int output_size ) :\
            Layer(input_size, output_size) {}
        // Return softmax conditional probabilities for given input.
        Eigen::MatrixXd forward_pass( const Eigen::MatrixXd & input ) const;
        // Return gradients of cross entropy loss function with respect to
        // layer weights, bias, and input.
        LayerGradients backward_pass( const Eigen::MatrixXd & input, const Eigen::MatrixXd & probs, const Eigen::MatrixXd & true_probs ) const;
};

// Neural network for classification.
// Uses cross entropy loss function.
class NeuralNet
{
    std::vector<Layer *> _layers;
    std::vector<LayerGradients> _gradients;
    NeuralNet();
    public:
        // There can be any number of hidden layers as long as input and output
        // dimensions match. The last layer should always of type Softmax.
        // Layers are joined in the same order as arguments:
        // NeuralNet( &h1, &h2, &softmax ) results in h1 -> h2 -> softmax.
        // All layers should have a base class of type Layer.
        template<typename... L>
        NeuralNet( L... layers ) : _layers { layers... } { _gradients.resize(_layers.size()); }
        // Perform forward pass and return probability of input belonging to each class.
        Eigen::MatrixXd probs( const Eigen::MatrixXd & input ) const;
        // Perform forward pass, backward pass, update _gradients and return
        // probability of input belonging to each class.
        Eigen::MatrixXd probs( const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs );
        const std::vector<LayerGradients> & gradients() const { return _gradients; }
        // Cross entropy loss.
        double loss( const Eigen::MatrixXd & probs, const Eigen::MatrixXd & true_probs ) const;
        // Update all parameters in all layers.
        void update( const std::vector<LayerUpdate> & updates );
        // Update one layer parameter.
        void update( const size_t layer_index, LayerParams lp, const Eigen::MatrixXd & update );
};

#endif
