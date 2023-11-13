#pragma once

#include <algorithm>
#include <random>
#include <tuple>
#include <vector>

using array2d = std::vector<std::vector<double>>;

std::tuple<array2d, array2d, array2d, array2d> train_test_split(
    array2d X, array2d Y, double train_size = 0.7) {
  array2d X_train, X_test, Y_train, Y_test;

  size_t n = X.size();
  size_t train_samples = n * train_size;

  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  size_t i;
  for (i = 0; i < train_samples; i++) {
    X_train.push_back(X[indices[i]]);
    Y_train.push_back(Y[indices[i]]);
  }

  for (; i < n; i++) {
    X_test.push_back(X[indices[i]]);
    Y_test.push_back(Y[indices[i]]);
  }

  return make_tuple(X_train, X_test, Y_train, Y_test);
}