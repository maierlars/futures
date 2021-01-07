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
  ASSERT_EQ(result, 24);
}

TEST(FutureTests, simple_abandon) {
  auto&& [future, promise] = futures::make_promise<int>();

  int result;
  std::move(future).finally([&](int x) noexcept {
    result = x;
  });

  ASSERT_DEATH(std::move(promise).abandon(), "");
  std::move(promise).fulfill(12);
  ASSERT_EQ(result, 12);
}

struct CollectTest {};

TEST(CollectTest, collect_vector) {

  bool reached = false;
  auto&& [f1, p1] = futures::make_promise<int>();
  auto&& [f2, p2] = futures::make_promise<int>();

  auto v = std::vector<future<int>>{};
  v.emplace_back(std::move(f1));
  v.emplace_back(std::move(f2));

  collect(v.begin(), v.end()).finally([&](auto&& x) noexcept {
    ASSERT_EQ(x.size(), 2);
    ASSERT_EQ(x[0], 1);
    ASSERT_EQ(x[1], 2);
    reached = true;
  });

  std::move(p2).fulfill(2);
  std::move(p1).fulfill(1);
  ASSERT_TRUE(reached);
}

TEST(CollectTest, collect_tuple) {

  bool reached = false;
  auto&& [f1, p1] = futures::make_promise<int>();
  auto&& [f2, p2] = futures::make_promise<int>();

  collect(std::move(f1), std::move(f2)).finally([&](auto&& x) noexcept {
    ASSERT_EQ(std::get<0>(x), 1);
    ASSERT_EQ(std::get<1>(x), 2);
    reached = true;
  });

  std::move(p2).fulfill(2);
  std::move(p1).fulfill(1);
  ASSERT_TRUE(reached);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
