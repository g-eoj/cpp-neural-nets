// Train a two layer neural network classifier.

#include <ctime>
#include <Eigen/Core>
#include <iostream>
#include <random>

#include "nn.h"
#include "two_layer_net.h"
#include "utils.h"

// TODO train/val split
// TODO batch size
// TODO random training example order

int main()
{
    srand( time(NULL) );

    // load and prep data
    std::string path = "wine.data"; // https://archive.ics.uci.edu/ml/datasets/wine
    Eigen::MatrixXd csv = LoadCSV<Eigen::MatrixXd>(path);
    Eigen::MatrixXd x = csv.rightCols(csv.cols() - 1);
    Eigen::MatrixXd y = OneHot(csv.col(0).cast<int>());

    // scale features to be between -1 and 1,
    // should put this logic in a class so it can be reused to scale
    // a validation/test set
    Eigen::RowVectorXd x_feature_maxs = x.colwise().maxCoeff();
    Eigen::RowVectorXd x_feature_mins = x.colwise().minCoeff();
    x = (x.rowwise() - x_feature_mins).array().rowwise() / (x_feature_maxs - x_feature_mins).array();
    x = (2 * x).array() - 1;

    // define network
    Hidden h1(x.cols(), 4);
    Softmax softmax(4, y.cols());
    TwoLayerNet net( h1, softmax );

    // for helper params
    ForwardPass fp;
    BackwardPass bp;
    Updates u;

    // training
    float lr = 0.8;
    int iterations = 100;
    double loss;
    double acc;
    Eigen::VectorXi pred;
    for ( int i = 1; i <= iterations; ++i )
    {
        // store outputs in fp
        loss = net.loss(x, y, fp);
        if ( !(i % 10) )
        {
            pred = Predict(fp.probs).array() + 1; //
            acc = Accuracy(csv.col(0).cast<int>(), pred);
            std::cout.width(5);
            std::cout << i;
            std::cout << " | loss: " << loss;
            std::cout << " | acc: " << acc;
            std::cout << std::endl;
            std::cout.width();
        }

        // calculate gradients, store in bp
        net.backpass(x, y, fp, bp);

//        std::cout << "Numerical/Analytical Gradient Relative Error" << std::endl;
//        std::cout << "W1: " << GradCheck(NetParams::W1, bp.dW1, net, x, y) << std::endl;
//        std::cout << "b1: " << GradCheck(NetParams::b1, bp.db1, net, x, y) << std::endl;
//        std::cout << "W2: " << GradCheck(NetParams::W2, bp.dW2, net, x, y) << std::endl;
//        std::cout << "b2: " << GradCheck(NetParams::b2, bp.db2, net, x, y) << std::endl;

        // SGD update
        u.W1 = bp.dW1 * -lr;
        u.b1 = bp.db1 * -lr;
        u.W2 = bp.dW2 * -lr;
        u.b2 = bp.db2 * -lr;
        net.update(u);
    }
}
