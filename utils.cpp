#include "utils.h"

// ---Preprocessing---

Batcher::Batcher( const unsigned int batch_size,
                  Eigen::MatrixXd & X, Eigen::MatrixXd & y, bool shuffle ) :\
                 _batch_size(batch_size), _batch_begin(0), _X(X), _y(y), _shuffle(shuffle)
{
    if ( batch_size >= y.rows() )
    {
        throw std::runtime_error("batch_size should be less than y.rows().");
    }
};

void Batcher::batch( Eigen::MatrixXd & X_batch, Eigen::MatrixXd & y_batch )
{
    if ( _batch_begin + _batch_size > _y.rows() ) {
        X_batch = _X.bottomRows(_y.rows() - _batch_begin);
        y_batch = _y.bottomRows(_y.rows() - _batch_begin);
        _batch_begin = 0;
        if ( _shuffle )
        {
            size_t random_seed = rand();
            ShuffleRows(_X, random_seed);
            ShuffleRows(_y, random_seed);
        }
    } else {
        X_batch = _X.block(_batch_begin, 0, _batch_size, _X.cols());
        y_batch = _y.block(_batch_begin, 0, _batch_size, _y.cols());
        _batch_begin += _batch_size;
    }
}

void MinMaxScaler::fit ( const Eigen::MatrixXd & data )
{
    _data_mins = data.colwise().maxCoeff();
    _data_maxs = data.colwise().minCoeff();
    _fitted = true;
}

void MinMaxScaler::transform ( Eigen::MatrixXd & data ) const
{
    if ( ! _fitted )
    {
        throw std::runtime_error("MinMaxScaler needs to be fitted before transform.");
    }
    data = (data.rowwise() - _data_mins).array().rowwise() /\
        (_data_maxs - _data_mins).array();
    data = data.array() * (_max - _min) + _min;
}

Eigen::MatrixXd OneHot( const Eigen::VectorXi & labels )
{
    unsigned int min_label = labels.minCoeff();
    unsigned int numLabels = 1 + labels.maxCoeff() - min_label;
    Eigen::MatrixXd oh = Eigen::MatrixXd::Zero(labels.size(), numLabels);
    for ( size_t l = 0; l < labels.size(); ++l )
    {
        oh(l, labels(l) - min_label) = 1;
    }
    return oh;
}

void ShuffleRows( Eigen::MatrixXd & matrix, const unsigned int random_seed )
{
    Eigen::PermutationMatrix<Eigen::Dynamic> permutation(matrix.rows());
    permutation.setIdentity();
    srand(random_seed);
    std::random_shuffle(permutation.indices().data(),\
                        permutation.indices().data()+permutation.indices().size());
    matrix = permutation * matrix;
}

void TrainTestSplit( Eigen::MatrixXd X, Eigen::MatrixXd y,\
                     Eigen::MatrixXd & X_train, Eigen::MatrixXd & X_test,\
                     Eigen::MatrixXd & y_train, Eigen::MatrixXd & y_test,\
                     float test_prop, bool shuffle, const unsigned int random_seed )
{
    if ( shuffle )
    {
        ShuffleRows(X, random_seed);
        ShuffleRows(y, random_seed);
    }
    size_t test_size = test_prop * y.rows();
    size_t train_size = y.rows() - test_size;
    X_train = X.topRows(train_size);
    X_test = X.bottomRows(test_size);
    y_train = y.topRows(train_size);
    y_test = y.bottomRows(test_size);
}

// ---End Preprocessing---

// ---Evaluation---

double Accuracy( const Eigen::VectorXi & yTrue, const Eigen::VectorXi & yPred )
{
    return (double)((yTrue - yPred).array() == 0).count() / yPred.rows();
}

double GradCheck( NeuralNet & net, const size_t layer_index, LayerParams lp,\
                  const Eigen::MatrixXd & input, const Eigen::MatrixXd & true_probs,
                  const size_t random_seed, const double & h )
{
    const LayerGradients & analytic_grads = net.gradients().at(layer_index);
    const size_t rc = ( lp == LayerParams::WEIGHTS ) ? analytic_grads.W.rows() : analytic_grads.b.size();
    const size_t cc = ( lp == LayerParams::WEIGHTS ) ? analytic_grads.W.cols() : 1;
    Eigen::MatrixXd numeric_grad(rc, cc);
    Eigen::MatrixXd h_matrix = Eigen::MatrixXd::Zero(rc, cc);
    double lp_plus_h;
    double lp_minus_h;
    for ( size_t i = 0; i < rc * cc; ++i )
    {
        h_matrix(i) = h;
        net.update(layer_index, lp, h_matrix);
        srand(random_seed);
        lp_plus_h = net.loss(net.probs(input, true), true_probs);

        h_matrix(i) = -h * 2;
        net.update(layer_index, lp, h_matrix);
        srand(random_seed);
        lp_minus_h = net.loss(net.probs(input, true), true_probs);

        h_matrix(i) = h;
        net.update(layer_index, lp, h_matrix);

        h_matrix(i) = 0;

        numeric_grad(i) = (lp_plus_h - lp_minus_h) / (2 * h);
    }
    return ( lp == LayerParams::WEIGHTS ) ? RelativeError(numeric_grad, analytic_grads.W) : RelativeError(numeric_grad, analytic_grads.b);
}

Eigen::VectorXi Predict( const Eigen::MatrixXd & probs )
{
    Eigen::VectorXi Predictions(probs.rows());
    for ( size_t r = 0; r < probs.rows(); ++r )
    {
        probs.row(r).maxCoeff(&Predictions(r));
    }
    return Predictions;
}

void PrintTrainingMetrics( const NeuralNet & net, const size_t & iteration,
                           const Eigen::MatrixXd & X_train, const Eigen::MatrixXd & X_val,
                           const Eigen::MatrixXd & y_train, const Eigen::MatrixXd & y_val )
{
    Eigen::MatrixXd probs_train = net.probs(X_train);
    Eigen::MatrixXd probs_val = net.probs(X_val);
    std::cout << std::setw(5) << iteration;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << " | loss: " << net.loss(probs_train, y_train);
    std::cout << " | acc: ";
    std::cout << Accuracy(Predict(y_train), Predict(probs_train));
    std::cout << " | val_loss: " << net.loss(probs_val, y_val);
    std::cout << " | val_acc: ";
    std::cout << Accuracy(Predict(y_val), Predict(probs_val));
    std::cout << std::endl;
}

void RandomData( size_t num_examples, size_t num_features, size_t num_classes,\
                 Eigen::MatrixXd & X, Eigen::VectorXi & y )
{
    X = Eigen::MatrixXd::Random(num_examples, num_features).array().abs();
    y = (Eigen::VectorXf::Random(num_examples).array().abs() * num_classes).cast<int>();
}

double RelativeError( const Eigen::MatrixXd & M1, const Eigen::MatrixXd & M2, const double & epsilon )
{
    return ( (M1 - M2).array().abs() /\
             (M1.array().abs() + M2.array().abs() + epsilon).maxCoeff() ).maxCoeff();
}

// ---End Evaluation---
