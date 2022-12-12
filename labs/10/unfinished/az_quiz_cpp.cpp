#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "az_quiz.h"
#include "az_quiz_mcts.h"

namespace py = pybind11;

class AZQuizCpp {
 public:
  template <typename T> using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

  static np_array<float> mcts(np_array<int8_t> board,
                              const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& network,
                              int num_simulations, double epsilon, double alpha) {
    AZQuiz game;
    std::copy_n(board.data(), game.board.size(), game.board.begin());

    Evaluator evaluator = [&network] (const AZQuiz& game, Policy& policy, float& value) {
      auto board = np_array<float>({1, game.N, game.N, game.C});
      game.representation(board.mutable_data());
      auto [np_policy, np_value] = network(board);
      std::copy_n(np_policy.data(), policy.size(), policy.begin());
      value = *np_value.data();
    };

    Policy policy;
    ::mcts(game, evaluator, num_simulations, epsilon, alpha, policy);

    return np_array<float>(policy.size(), policy.data());
  }

  static void simulated_games_start(const std::function<std::pair<np_array<float>, np_array<float>>(np_array<float> boards)>& network,
                                    int threads, int num_simulations, int sampling_moves, double epsilon, double alpha) {
  }

  static py::list simulated_game() {
    return py::list();
  }

  static void simulated_games_stop() {
  }
};


PYBIND11_MODULE(az_quiz_cpp, m) {
    m.doc() = "AZQuiz C++ Module";

    m.def("mcts", &AZQuizCpp::mcts, "Run a Monte Carlo Tree Search");
    m.def("simulated_games_start", &AZQuizCpp::simulated_games_start, "Start generating simulated games");
    m.def("simulated_game", &AZQuizCpp::simulated_game, "Get one simulated game");
    m.def("simulated_games_stop", &AZQuizCpp::simulated_games_start, "Stop generating simulated games");
}
