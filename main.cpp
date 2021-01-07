#include <cassert>
#include <iostream>
#include <thread>

#include <futures/completion-queue.h>
#include <futures/futures.h>
#include <futures/utilities.h>

#include "test/test-helper.h"

#include <gtest/gtest.h>

using namespace futures;

struct FutureTests {};

TEST(FutureTests, simple_test) {
  auto&& [future, guard] = make_delayed_fulfilled<int>(12);

  int result;
  std::move(future).and_then([](int x) noexcept { return 2 * x; }).finally([&](int x) noexcept {
    result = x;
  });

  std::move(guard).trigger();
  EXPECT_EQ(result, 24);
}

TEST(FutureTests, simple_abandon) {
  auto&& [future, promise] = futures::make_promise<int>();

  int result;
  std::move(future).finally([&](int x) noexcept { result = x; });

  EXPECT_DEATH(std::move(promise).abandon(), "");
  std::move(promise).fulfill(12);
  EXPECT_EQ(result, 12);
}

struct CollectTest {};

TEST(CollectTest, collect_vector) {
  bool reached = false;
  auto&& [f1, p1] = futures::make_promise<int>();
  auto&& [f2, p2] = futures::make_promise<int>();

  auto fs = std::vector<future<int>>{};
  auto ps = std::vector<promise<int>>{};

  const auto number_of_futures = 4;

  for (size_t i = 0; i < number_of_futures; i++) {
    auto&& [f, p] = futures::make_promise<int>();
    fs.emplace_back(std::move(f).and_then([i](int x) noexcept { return i * x; }));
    ps.emplace_back(std::move(p));
  }

  collect(fs.begin(), fs.end()).finally([&](auto&& xs) noexcept {
    EXPECT_EQ(xs.size(), number_of_futures);
    for (size_t i = 0; i < number_of_futures; i++) {
      EXPECT_EQ(xs[i], i);
    }
    reached = true;
  });

  for (auto&& p : ps) {
    std::move(p).fulfill(1);
  }
  EXPECT_TRUE(reached);
}

TEST(CollectTest, collect_tuple) {
  bool reached = false;
  auto&& [f1, p1] = futures::make_promise<int>();
  auto&& [f2, p2] = futures::make_promise<int>();

  collect(std::move(f1), std::move(f2)).finally([&](auto&& x) noexcept {
    EXPECT_EQ(std::get<0>(x), 1);
    EXPECT_EQ(std::get<1>(x), 2);
    reached = true;
  });

  std::move(p2).fulfill(2);
  std::move(p1).fulfill(1);
  EXPECT_TRUE(reached);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
