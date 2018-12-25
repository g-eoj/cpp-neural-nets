// Train a neural network classifier.

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

#include "nn.h"
#include "utils.h"

// TODO batch size
// TODO random training example order
// TODO Optimizer class
// TODO dropout
// TODO random seeds

int main()
{
    std::string path = "wine.data"; // https://archive.ics.uci.edu/ml/datasets/wine
    Eigen::MatrixXd csv = LoadCSV<Eigen::MatrixXd>(path);

    // train/validation shuffle split
    ShuffleRows(csv);
    size_t val_size = 0.3 * csv.rows();
    Eigen::MatrixXd x_train = csv.block(val_size, 1, csv.rows() - val_size, csv.cols() - 1);
    Eigen::MatrixXd x_val = csv.block(0, 1, val_size, csv.cols() - 1);
    Eigen::MatrixXd y_train = OneHot(csv.block(val_size, 0, csv.rows() - val_size, 1).cast<int>());
    Eigen::MatrixXd y_val = OneHot(csv.block(0, 0, val_size, 1).cast<int>());
    std::cout << "Train set size: " << y_train.rows() << std::endl;
    std::cout << "  Val set size: " << y_val.rows() << std::endl;
    std::cout << std::endl;

    // scale features to be between -1 and 1
    MinMaxScaler scaler(-1, 1);
    scaler.fit(x_train);
    scaler.transform(x_train);
    scaler.transform(x_val);

    // define network
    srand(time(NULL));
    Hidden h1(x_train.cols(), 6);
    Hidden h2(6, 6);
    Softmax softmax(6, y_train.cols());
    NeuralNet net( &h1, &h2, &softmax );

    // train network
    int iterations = 100;
    float lr = 0.2;
    Eigen::MatrixXd probs_train;
    Eigen::MatrixXd probs_val;
    for ( int i = 1; i <= iterations; ++i )
    {
        probs_train = net.probs(x_train, y_train);
        if ( !(i % 10) )
        {
            probs_val = net.probs(x_val);
            // metrics
            std::cout << std::setw(5) << i;
            std::cout << std::fixed << std::setprecision(5);
            std::cout << " | loss: " << net.loss(probs_train, y_train);
            std::cout << " | acc: ";
            std::cout << Accuracy(Predict(y_train), Predict(probs_train));
            std::cout << " | val_loss: " << net.loss(probs_val, y_val);
            std::cout << " | val_acc: ";
            std::cout << Accuracy(Predict(y_val), Predict(probs_val));
            std::cout << std::endl;
        }

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
    probs_val = net.probs(x_val);
    std::cout << "\nFinal Validation Accuracy: " ;
    std::cout << Accuracy(Predict(y_val), Predict(probs_val));
    std::cout << std::endl;
}
