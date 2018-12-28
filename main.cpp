// Train a neural network classifier.

#include <Eigen/Core>
#include <iomanip>
#include <iostream>

#include "nn.h"
#include "optimizers.h"
#include "utils.h"

// TODO batch size
// TODO dropout

int main()
{
    std::string path = "wine.data"; // https://archive.ics.uci.edu/ml/datasets/wine
    Eigen::MatrixXd csv = LoadCSV<Eigen::MatrixXd>(path);

    // train/validation shuffle split
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd x_val;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd y_val;
    TrainTestSplit(csv.rightCols(csv.cols() - 1), OneHot(csv.leftCols(1).cast<int>()),\
                     x_train, x_val, y_train, y_val, 0.3);
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
    Hidden h1(x_train.cols(), 10);
    Softmax softmax(10, y_train.cols());
    NeuralNet net( &h1, &softmax );

    // train network
    size_t iterations = 100;
    Eigen::MatrixXd probs_train;
    Eigen::MatrixXd probs_val;
    SGD sgd(net, 0.1, 0.5);
    for ( size_t i = 1; i <= iterations; ++i )
    {
        sgd.fit(x_train, y_train);
        if ( !(i % 10) )
        {
            probs_train = net.probs(x_train);
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
        if ( i == iterations / 2 )
        {
            sgd.lr(sgd.lr() / 2);
            sgd.momentum(0.9);
        }
    }
    probs_val = net.probs(x_val);
    std::cout << "\nFinal Validation Accuracy: " ;
    std::cout << Accuracy(Predict(y_val), Predict(probs_val));
    std::cout << std::endl;
}
