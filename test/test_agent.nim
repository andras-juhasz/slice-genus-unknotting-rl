import std/unittest
import std/options
import std/sequtils
import std/math

import "../spherogram-nim/src/links"

import ../src/agent
import ../src/dataset

test "newRandomWalkParam":
    expect ValueError:
        discard newRandomWalkParam(1.0, 1.0, 5, 25, "other")

test "band_operations_random_walk":
    var params = random_walk_param_default("unknotting")
    var dataset = dataset_read_csv("../datasets/rolfsen.csv")[0 ..< 35]
    check abs(dataset.evaluate_invariant("unknotting", band_operations_random_walk("unknotting", params, dataset.links, 20000)).get() - 1.0) < 1e-6

    dataset = dataset_read_csv("../datasets/linkinfo-to-9.csv")[0 ..< 25]
    check abs(dataset.evaluate_invariant("weak splitting", band_operations_random_walk("weak splitting", params, dataset.links, 20000)).get() - 1.0) < 1e-6

    params = random_walk_param_default("slice")
    dataset = dataset_read_csv("../datasets/rolfsen.csv")[0 ..< 15]
    check abs(dataset.evaluate_invariant("slice", band_operations_random_walk("slice", params, dataset.links, 20000)).get() - 1.0) < 1e-6

    dataset = dataset_read_csv("../datasets/linkinfo-to-9.csv")[0 ..< 20]
    check abs(dataset.evaluate_invariant("slice", band_operations_random_walk("slice", params, dataset.links, 20000)).get() - 1.0) < 1e-6
