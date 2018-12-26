#include <Eigen/Core>
#include <vector>

#include "nn.h"

class Optimizer
{
    Optimizer();
protected:
    double _lr;
    std::vector<LayerUpdate> _update;
    Optimizer( const NeuralNet & net, const double lr ) : _lr(lr)
    {
        LayerUpdate layer_update;
        for ( auto layer = net._layers.begin(); layer != net._layers.end(); ++layer )
        {
            layer_update.W.resizeLike((**layer).W());
            layer_update.W.setZero();
            layer_update.b.resizeLike((**layer).b());
            layer_update.b.setZero();
            _update.push_back(layer_update);
        }
    }
    virtual void _calculate_update( const size_t layer_index, const LayerGradients & gradients ) = 0;
public:
    virtual ~Optimizer() {}
    void fit( NeuralNet & net, const Eigen::MatrixXd & X, const Eigen::MatrixXd & y )
    {
        net.gradients(X, y);
        for ( size_t layer_index = 0; layer_index < net.gradients().size(); ++layer_index)
        {
            _calculate_update(layer_index, net.gradients().at(layer_index));
        }
        net.update(_update);
    }
    const double & lr() const { return _lr; }
    void lr( const double & lr ) { _lr = lr; }
};

class SGD : public Optimizer {
    double _momentum;
    void _calculate_update( const size_t layer_index, const LayerGradients & gradients )
    {
        _update.at(layer_index).W = _momentum * _update.at(layer_index).W - _lr * gradients.W;
        _update.at(layer_index).b = _momentum * _update.at(layer_index).b - _lr * gradients.b;
    }
public:
    SGD( const NeuralNet & net, const double lr=0.01, const double momentum=0.0 )\
        : Optimizer(net, lr), _momentum(momentum) {}
    const double & momentum() const { return _momentum; }
    void momentum( const double & momentum ) { _momentum = momentum; }
};
