#include <ctime>
#include <Eigen/Core>
#include <iostream>
#include <random>
#include <vector>

#include "nn.h"
#include "utils.h"

class Test
{
    unsigned int _pass_count;
    unsigned int _fail_count;
    const char * _test_type = nullptr;
    bool _verbose_tests = false;
public:
    Test( const char * test_type )
    {
        reset(test_type);
    }
    void reset( const char * test_type )
    {
        _pass_count = 0;
        _fail_count = 0;
        _test_type = test_type;
    }
    unsigned int pass_count() const { return _pass_count; }
    unsigned int fail_count() const { return _fail_count; }
    void verbose( const bool flag ) { _verbose_tests = flag; }
    void run( const char * description, const int flag)
    {
        const char * pf = nullptr;
        if (flag) {
            ++_pass_count;
            pf = "pass";
        } else {
            ++_fail_count;
            pf = "fail";
        }
        if (_verbose_tests) {
        std::cout << _test_type << ": " << description << " -> " << pf << std::endl;
        }
    }
    void report() const
    {
        std::cout << _test_type << " Summary" << std::endl;
        std::cout << "Passed: " << _pass_count;
        std::cout << " | Failed: " << _fail_count;
        std::cout << std::endl << std::endl;
    }
};

void random_data( size_t num_examples, size_t num_features, size_t num_classes,\
                  Eigen::MatrixXd & X, Eigen::VectorXi & y )
{
    X.setRandom(num_examples, num_features);
    y = (Eigen::VectorXf::Random(num_examples).array().abs() * num_classes).cast<int>();
}

int main()
{
    srand(time(NULL));
    unsigned int total_passed = 0;
    unsigned int total_failed = 0;

    // Util tests
    Test test("Util Functions");
    {
        // OneHot
        Eigen::VectorXi labels(5);
        Eigen::MatrixXd labels_oh(5, 3);
        labels_oh << 1,0,0,
                     0,0,1,
                     0,1,0,
                     0,1,0,
                     1,0,0;
        labels << 1, 3, 2, 2, 1;
        test.run("OneHot One Based", OneHot(labels) == labels_oh);
        labels << 0, 2, 1, 1, 0;
        test.run("OneHot Zero Based", OneHot(labels) == labels_oh);
        // Predict
        test.run("Predict", Predict(labels_oh) == labels);
        // Accuracy
        Eigen::VectorXi preds(5);
        preds << 0, 2, 0, 2, 0;
        test.run("Accuracy", abs(Accuracy(labels, preds) - 3 / 5) < 1e-16);
        // ShuffleRows
        Eigen::MatrixXd M = Eigen::MatrixXd::Random(7, 5);
        Eigen::MatrixXd M_shuf = M;
        Eigen::MatrixXd M_shuf_2 = M;
        ShuffleRows(M_shuf);
        ShuffleRows(M_shuf_2, 1);
        test.run("ShuffleRows", ! M.isApprox(M_shuf) && M_shuf.isApprox(M_shuf_2) &&\
                 M.colwise().sum().isApprox(M_shuf.colwise().sum()));
        // MinMaxScaler
        MinMaxScaler default_scaler;
        default_scaler.fit(M);
        default_scaler.transform(M);
        test.run("MinMaxScaler Default Range", (M.minCoeff() == 0) && (M.maxCoeff() == 1));
        MinMaxScaler scaler(-2, 2);
        scaler.fit(M_shuf);
        scaler.transform(M_shuf);
        test.run("MinMaxScaler [-2, 2]", (M_shuf.minCoeff() == -2) && (M_shuf.maxCoeff() == 2));
        // TrainTestSplit
        Eigen::MatrixXd X_train;
        Eigen::MatrixXd X_test;
        Eigen::MatrixXd y_train;
        Eigen::MatrixXd y_test;
        unsigned int test_size = 0.3 * M.rows();
        TrainTestSplit(M, M, X_train, X_test, y_train, y_test, 0.3);
        test.run("TrainTestSplit", (X_train == y_train) && (X_test == y_test) &&\
                 (X_train.rows() == M.rows() - test_size) && (X_test.rows() == test_size));
        // Batcher
        Batcher batcher(2, M, M);
        Eigen::MatrixXd x_batch;
        Eigen::MatrixXd y_batch;
        for ( size_t i = 0; i <= M.rows(); ++i )
        {
            batcher.batch(x_batch, y_batch);
        }
        bool test_1 = (M.bottomRows(1) == x_batch) && (x_batch == y_batch);
        batcher.batch(x_batch, y_batch);
        bool test_2 = (M.topRows(2) == x_batch) && (x_batch == y_batch);
        test.run("Batcher", test_1 && test_2);
    }
    total_passed += test.pass_count();
    total_failed += test.fail_count();
    test.report();
    // End util tests

    // Layer tests with fixed data
    test.reset("Layers");
    {
        Eigen::Matrix<double, 1, 2> x;
        Eigen::Matrix<double, 1, 3> y;
        Eigen::Matrix<double, 2, 2> W1;
        Eigen::Matrix<double, 2, 1> b1;
        Eigen::Matrix<double, 2, 3> W2;
        Eigen::Matrix<double, 2, 2> W1_grad;
        Eigen::Matrix<double, 2, 1> b1_grad;
        Eigen::Matrix<double, 1, 2> h1_grad;
        Eigen::Matrix<double, 2, 3> W2_grad;
        Eigen::Matrix<double, 3, 1> b2_grad;
        Eigen::Matrix<double, 1, 2> h1_out;
        Eigen::Matrix<double, 1, 3> softmax_out;
        x << 16, -6;
        y << 0, 0, 1;
        W1 << 0.18, 0.04, 0.1, 0.22;
        b1 << 0.5, 0.1;
        W2 << 0.19, -0.1, 0.1, -0.02, -0.01, 0.04;
        h1_out << 2.28, 0;
        softmax_out << 0.4290527237144287,
                       0.2214905246525316,
                       0.3494567516330398;
        W1_grad << -0.0909337567393238, 0, 0.034100158777246425, 0;
        b1_grad << -0.005683359796207738, 0;
        h1_grad << -0.005683359796207738, -0.0368176896554923;
        W2_grad << 0.9782402100688974, 0.504998396207772, -1.4832386062766691, 0, 0, 0;
        b2_grad << 0.4290527237144287, 0.2214905246525316, -0.6505432483669602;
        Hidden h1(2, 2);
        Softmax softmax(2, 3);
        h1.update(LayerParams::WEIGHTS, W1 - h1.W());
        test.run("Layer Weight Update", W1.isApprox(h1.W()));
        h1.update(LayerParams::BIAS, b1);
        test.run("Layer Bias Update", b1.isApprox(h1.b()));
        h1.update(LayerParams::BIAS, -b1);
        softmax.update(LayerParams::WEIGHTS, W2 - softmax.W());
        test.run("Hidden Forward Pass", h1.forward_pass(x).isApprox(h1_out));
        test.run("Softmax Forward Pass", softmax.forward_pass(h1_out).isApprox(softmax_out));
        LayerGradients softmax_grads = softmax.backward_pass( h1_out, softmax_out, y);
        test.run("Softmax Backward Pass", softmax_grads.W.isApprox(W2_grad) &&\
                                          softmax_grads.b.isApprox(b2_grad) &&\
                                          softmax_grads.input.isApprox(h1_grad));
        LayerGradients h1_grads = h1.backward_pass(x, h1_out, h1_grad);
        test.run("Hidden Backward Pass", h1_grads.W.isApprox(W1_grad) &&\
                                         h1_grads.b.isApprox(b1_grad));
    }
    total_passed += test.pass_count();
    total_failed += test.fail_count();
    test.report();
    // End layer tests with fixed data

    // Two layer net with fixed data
    test.reset("Two Layer Fixed Data");
    {
        Eigen::Matrix<double, 1, 2> x;
        Eigen::Matrix<double, 1, 3> y;
        Eigen::Matrix<double, 2, 2> W1;
        Eigen::Matrix<double, 2, 1> b1;
        Eigen::Matrix<double, 2, 3> W2;
        Eigen::Matrix<double, 3, 1> b2;
        Eigen::Matrix<double, 2, 2> W1_grad;
        Eigen::Matrix<double, 2, 1> b1_grad;
        Eigen::Matrix<double, 2, 3> W2_grad;
        Eigen::Matrix<double, 3, 1> b2_grad;
        Eigen::Matrix<double, 1, 3> softmax_out;
        double loss = 1.0513754685043635;
        x << 16, -6;
        y << 0, 0, 1;
        W1 << 0.18, 0.04, 0.1, 0.22;
        b1 << 0.5, -0.2;
        W2 << 0.19, -0.1, 0.1, -0.02, -0.01, 0.04;
        b2 << 0, 0, 0;
        softmax_out << 0.4290527237144287,
                       0.2214905246525316,
                       0.3494567516330398;
        W1_grad << -0.0909337567393238, 0, 0.034100158777246425, 0;
        b1_grad << -0.005683359796207738, 0;
        W2_grad << 0.9782402100688974, 0.504998396207772, -1.4832386062766691, 0, 0, 0;
        b2_grad << 0.4290527237144287, 0.2214905246525316, -0.6505432483669602;
        Hidden h1(2, 2);
        Softmax softmax(2, 3);
        NeuralNet net( &h1, &softmax );
        LayerUpdate h1_update;
        h1_update.W = W1 - h1.W();
        h1_update.b = b1 - h1.b();
        LayerUpdate softmax_update;
        softmax_update.W = W2 - softmax.W();
        softmax_update.b = b2 - softmax.b();
        std::vector<LayerUpdate> updates = {h1_update, softmax_update};
        net.update(updates);
        test.run("Update All Layers", h1.W().isApprox(W1) &&\
                                      h1.b().isApprox(b1) &&\
                                      softmax.W().isApprox(W2) &&\
                                      softmax.b().isApprox(b2));
        net.update(0, LayerParams::BIAS, -b1);
        b1 << 0, 0;
        test.run("Update Single Parameter", h1.b().isApprox(b1));
        test.run("Probs", net.probs(x).isApprox(softmax_out));
        test.run("Loss", abs(net.loss(softmax_out, y) - loss) < 1e-16);
        net.gradients(x, y);
        test.run("Gradients", net.gradients().at(0).W.isApprox(W1_grad) &&\
                              net.gradients().at(0).b.isApprox(b1_grad) &&\
                              net.gradients().at(1).W.isApprox(W2_grad) &&\
                              net.gradients().at(1).b.isApprox(b2_grad));
    }
    total_passed += test.pass_count();
    total_failed += test.fail_count();
    test.report();    //
    // End two layer net with fixed data

    // Two layer net with random data
    test.reset("Two Layer Random Data");
    {
        size_t num_examples = 30;
        size_t num_features = 10;
        size_t num_classes = 10;
        size_t hidden_size = 10;
        Eigen::MatrixXd X;
        Eigen::VectorXi y;
        random_data(num_examples, num_features, num_classes, X, y);
        Eigen::MatrixXd y_oh = OneHot(y);
        Hidden h1(X.cols(), hidden_size);
        Softmax softmax(hidden_size, y_oh.cols());
        NeuralNet two_layer_net( &h1, &softmax );
        test.run("Initial Loss", two_layer_net.loss(two_layer_net.probs(X), y_oh) + log(1. / num_classes) < 0.2);
        double epsilon = 1e-08;
        two_layer_net.gradients(X, y_oh);
        test.run("Gradient Check W1", GradCheck(two_layer_net, 0, LayerParams::WEIGHTS, X, y_oh) < epsilon);
        test.run("Gradient Check b1", GradCheck(two_layer_net, 0, LayerParams::BIAS, X, y_oh) < epsilon);
        test.run("Gradient Check W2", GradCheck(two_layer_net, 1, LayerParams::WEIGHTS, X, y_oh) < epsilon);
        test.run("Gradient Check b2", GradCheck(two_layer_net, 1, LayerParams::BIAS, X, y_oh) < epsilon);
    }
    total_passed += test.pass_count();
    total_failed += test.fail_count();
    test.report();
    // End two layer net with random data

    std::cout << "Totals: ";
    std::cout << "Passed: " << total_passed << " | Failed: " << total_failed << std::endl;
}
