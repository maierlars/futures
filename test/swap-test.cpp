#include "test-helper.h"

struct SwapTests : testing::Test {
};


TEST_F(SwapTests, swap_promise) {

  signal_marker reached_last_1{"reached-last-1"};
  signal_marker reached_last_2{"reached-last-2"};

  auto&& [f1, p1] = make_promise<int>();
  auto&& [f2, p2] = make_promise<int>();

  using std::swap;
  swap(p1, p2);

  std::move(f1).finally([&](int x) noexcept {
    EXPECT_EQ(x, 2);
    reached_last_1.signal();
  });
  std::move(f2).finally([&](int x) noexcept {
    EXPECT_EQ(x, 1);
    reached_last_2.signal();
  });

  std::move(p1).fulfill(1);
  std::move(p2).fulfill(2);
}
