#pragma once

#include <array>
#include <functional>

#include "az_quiz.h"

typedef std::array<float, AZQuiz::ACTIONS> Policy;

typedef std::function<void(const AZQuiz&, Policy&, float&)> Evaluator;

void mcts(const AZQuiz& game, const Evaluator& evaluator, int num_simulations, double epsilon, double alpha, Policy& policy) {
  // TODO: Implement MCTS.
}
