// Train a neural network classifier.

#include <ctime>
#include <Eigen/Core>
#include <iomanip>
#include <iostream>
#include <random>

#include "nn.h"
#include "utils.h"

// TODO train/val split
// TODO batch size
// TODO random training example order
// TODO Optimizer class
// TODO backprop with true_probs that are not one hot
// TODO dropout

int main()
{
    srand( time(NULL) );

    // load and prep data
    std::string path = "wine.data"; // https://archive.ics.uci.edu/ml/datasets/wine
    Eigen::MatrixXd csv = LoadCSV<Eigen::MatrixXd>(path);
    Eigen::MatrixXd x = csv.rightCols(csv.cols() - 1);
    Eigen::MatrixXd y = OneHot(csv.col(0).cast<int>());

    // scale features to be between -1 and 1
    // TODO put this logic in a class so it can be reused to scale
    // a validation/test set
    Eigen::RowVectorXd x_feature_maxs = x.colwise().maxCoeff();
    Eigen::RowVectorXd x_feature_mins = x.colwise().minCoeff();
    x = (x.rowwise() - x_feature_mins).array().rowwise() /\
        (x_feature_maxs - x_feature_mins).array();
    x = (2 * x).array() - 1;

    // define network
    Hidden h1(x.cols(), 6);
    Hidden h2(6, 6);
    Softmax softmax(6, y.cols());
    NeuralNet net( &h1, &h2, &softmax );

    // train network
    int iterations = 100;
    float lr = 0.8;
    Eigen::MatrixXd probs;
    for ( int i = 1; i <= iterations; ++i )
    {
        probs = net.probs(x, y);
        if ( !(i % 10) )
        {
            // metrics
            std::cout << std::setw(5) << i;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << " | loss: " << net.loss(probs, y);
            std::cout << " | acc: ";
            std::cout << Accuracy(csv.col(0).cast<int>(), Predict(probs).array() + 1);
            std::cout << std::endl;
        }

//        std::cout << "Numerical/Analytical Gradient Relative Error" << std::endl;
//        for ( size_t layer_index = 0; layer_index < net.gradients().size(); ++layer_index )
//        {
//            std::cout << "W" << layer_index + 1 << ": ";
//            std::cout << GradCheck(net, layer_index, LayerParams::WEIGHTS,\
//                                   net.gradients().at(layer_index).W, x, y) << std::endl;
//            std::cout << "b" << layer_index + 1 << ": ";
//            std::cout << GradCheck(net, layer_index, LayerParams::BIAS,\
//                                   net.gradients().at(layer_index).b, x, y) << std::endl;
//        }

        // SGD
        std::vector<LayerUpdate> sgd_update;
        LayerUpdate sgd_layer_update;
        for ( auto layer_gradients = net.gradients().begin();
                   layer_gradients != net.gradients().end() ;
                   ++layer_gradients )
        {
            sgd_layer_update.W = layer_gradients->W * -lr;
            sgd_layer_update.b = layer_gradients->b * -lr;
            sgd_update.push_back(sgd_layer_update);
        }
        net.update(sgd_update);
    }
    // test forward pass
    probs = net.probs(x);
    std::cout << "\nFinal Accuracy: " ;
    std::cout << Accuracy(csv.col(0).cast<int>(), Predict(probs).array() + 1);
    std::cout << std::endl;
}
