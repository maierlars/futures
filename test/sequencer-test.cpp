#include <mellon/sequencer.h>
#include "test-helper.h"

struct SequencerTests {};


TEST(SequencerTests, simple_test) {
  auto [f1, p1] = make_promise<int>();
  auto [f2, p2] = make_promise<int>();

  bool p1_fulfilled = false;
  bool p2_fulfilled = false;
  bool then2_executed = false;

  auto f =
      mellon::sequence(std::move(f1))
          .append([&, f2 = std::move(f2)](int x) mutable noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_FALSE(p2_fulfilled);
            if (x != 12) {
              return std::move(f2);
            }
            return future<int>{std::in_place, 12};
          })
          .append([&](int x) noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_TRUE(p2_fulfilled);
            then2_executed = true;
            return future<int>{std::in_place, 78};
          })
          .compose();

  std::move(f).finally([](int&& x) noexcept {
    EXPECT_EQ(x, 78);
  });

  p1_fulfilled = true;
  std::move(p1).fulfill(13);
  p2_fulfilled = true;
  EXPECT_FALSE(then2_executed);
  std::move(p2).fulfill(37);
  EXPECT_TRUE(then2_executed);
}

TEST(SequencerTests, capture_test) {
  auto [f1, p1] = make_promise<int>();
  auto [f2, p2] = make_promise<int>();

  auto f =
      mellon::sequence(std::move(f1))
          .append_capture([&, f2 = std::move(f2)](int x) mutable {
            if (x == 12) {
              return std::move(f2);
            }
            throw std::runtime_error("some test error");
          })
          .then_do([&](int x) {
            ADD_FAILURE() << "This should never be executed";
            return future<int>{std::in_place, 78};
          })
          .compose();

  std::move(f).finally([](expect::expected<int>&& x) noexcept {
    EXPECT_TRUE(x.has_exception<std::runtime_error>());
  });

  std::move(p1).fulfill(13);
  std::move(p2).fulfill(37);
}

/*
TEST_F(SequencerTest, then_do_test) {
  auto [f1, p1] = make_promise<int>();
  auto [f2, p2] = make_promise<int>();

  bool p1_fulfilled = false;
  bool p2_fulfilled = false;
  bool then2_executed = false;

  auto f =
      mellon::sequence(std::move(f1))
          .then_do([&, f2 = std::move(f2)](int x) mutable noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_FALSE(p2_fulfilled);
            if (x != 12) {
              return mellon::mr<default_test_tag>(true, std::move(f2));
            }
            return mellon::mr<default_test_tag>(false, future<int>{std::in_place, 12});
          })
          .then_do([&](bool y, int x) noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_TRUE(p2_fulfilled);
            then2_executed = true;
            return future<std::tuple<int>>{std::in_place, 78};
          })
          .compose();

  std::move(f).finally_apply([](int&& x) noexcept {
    EXPECT_EQ(x, 78);
  });

  p1_fulfilled = true;
  std::move(p1).fulfill(13);
  p2_fulfilled = true;
  EXPECT_FALSE(then2_executed);
  std::move(p2).fulfill(37);
  EXPECT_TRUE(then2_executed);
}


TEST_F(SequencerTest, expected_test) {
  auto [f1, p1] = make_promise<int>();
  auto [f2, p2] = make_promise<int>();

  bool p1_fulfilled = false;
  bool p2_fulfilled = false;
  bool then2_executed = false;

  auto f =
      mellon::sequence(std::move(f1))
          .then_do([&, f2 = std::move(f2)](int x) mutable noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_FALSE(p2_fulfilled);
            if (x != 12) {
              return mellon::mr<default_test_tag>(true, std::move(f2));
            }
            return mellon::mr<default_test_tag>(false, future<int>{std::in_place, 12});
          })
          .then_do([&](bool y, int x) noexcept {
            EXPECT_TRUE(p1_fulfilled);
            EXPECT_TRUE(p2_fulfilled);
            then2_executed = true;
            return future<std::tuple<int>>{std::in_place, 78};
          })
          .compose();

  std::move(f).finally_apply([](int&& x) noexcept {
    EXPECT_EQ(x, 78);
  });

  p1_fulfilled = true;
  std::move(p1).fulfill(13);
  p2_fulfilled = true;
  EXPECT_FALSE(then2_executed);
  std::move(p2).fulfill(37);
  EXPECT_TRUE(then2_executed);
}
*/
