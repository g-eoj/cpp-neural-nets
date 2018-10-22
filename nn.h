// Classes for creating a neural network.

#ifndef _NN
#define _NN

#include <Eigen/Core>

enum class LayerParams { WEIGHTS, BIAS };

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
        const Eigen::MatrixXd & W() const { return _W; }
        const Eigen::VectorXd & b() const { return _b; }
        // add u to WEIGHTS or BIAS
        void update( LayerParams lp, const Eigen::MatrixXd & u );
};

// Dense layer with relu activation.
class Hidden : public Layer
{
    Hidden();
    public:
        Hidden( int input_size, int output_size ) :\
            Layer(input_size, output_size) {}
        // do a forward pass
        Eigen::MatrixXd output ( const Eigen::MatrixXd & input ) const;
};

// Dense layer with softmax activation.
class Softmax : public Layer
{
    Softmax();
    public:
        Softmax( int input_size, int output_size ) :\
            Layer(input_size, output_size) {}
        // do a forward pass
        Eigen::MatrixXd output( const Eigen::MatrixXd & input ) const;
};

#endif
