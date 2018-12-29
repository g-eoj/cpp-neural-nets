// Train a neural network classifier.

#include <Eigen/Core>
#include <iostream>

#include "nn.h"
#include "optimizers.h"
#include "utils.h"

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
    size_t batch_size = 16;
    size_t epochs = 10;
    size_t steps_per_epoch = (y_train.rows() + batch_size - 1) / batch_size;
    size_t epoch = 0;
    Eigen::MatrixXd x_batch;
    Eigen::MatrixXd y_batch;
    SGD sgd(net, 0.03, 0.5);
    Batcher batcher(batch_size, x_train, y_train);
    for ( size_t i = 1; i <= epochs * steps_per_epoch; ++i )
    {
        batcher.batch(x_batch, y_batch);
        sgd.fit(x_batch, y_batch);
        if ( !(i % steps_per_epoch) )
        {
            epoch += 1;
            PrintTrainingMetrics(net, epoch, x_train, x_val, y_train, y_val);
            if ( epoch == epochs / 2 )
            {
                sgd.lr(sgd.lr() / 2);
                sgd.momentum(0.9);
            }
        }
    }
    std::cout << "\nFinal Validation Accuracy: " ;
    std::cout << Accuracy(Predict(y_val), Predict(net.probs(x_val)));
    std::cout << std::endl;
}
