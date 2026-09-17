#include <Eigen/Core>

#include <iostream>

#include "a_star/a_star_search.h"

int main() {
  constexpr int kSize = 5;
  Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(kSize, kSize);
  Eigen::MatrixXd height = Eigen::MatrixXd::Zero(kSize, kSize);
  Eigen::MatrixXd elevation = Eigen::MatrixXd::Zero(kSize, kSize);

  // Split the map into two traversable components without making the goal
  // itself invalid. The middle search must exhaust the left component.
  cost.col(2).setConstant(100.0);

  Astar astar;
  astar.Init(20.0, 1, 1.0, 0.0, cost, height, elevation);

  const Eigen::Vector3i start(0, 0, 0);
  const Eigen::Vector3i reachable_goal(0, 1, 1);
  const Eigen::Vector3i unreachable_goal(0, 4, 4);

  if (!astar.Search(start, reachable_goal)) {
    std::cerr << "first reachable search failed" << std::endl;
    return 1;
  }
  if (astar.Search(start, unreachable_goal)) {
    std::cerr << "disconnected search unexpectedly succeeded" << std::endl;
    return 2;
  }
  if (!astar.Search(start, reachable_goal)) {
    std::cerr << "reachable search failed after a failed search" << std::endl;
    return 3;
  }

  std::cout << "A_STAR_REPEAT_SEARCH=PASS" << std::endl;
  return 0;
}
