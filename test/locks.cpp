#include "test-helper.h"

#include <mellon/locks.h>

TEST(locks, simple_test) {
  mellon::future_mutex<default_test_tag> mutex;
  bool executed = false;

  // this should execute immediately
  mutex.async_lock().finally(
      [&](std::unique_lock<mellon::future_mutex<default_test_tag>>&& lock) noexcept {
        executed = true;
        ASSERT_TRUE(mutex.is_locked());
      });

  ASSERT_FALSE(mutex.is_locked());
  ASSERT_TRUE(executed);
}

TEST(locks, async_lock_test) {
  mellon::future_mutex<default_test_tag> mutex;
  mutex.lock();

  ASSERT_TRUE(mutex.is_locked());

  std::atomic<bool> executed = false;
  // this should not execute immediately
  auto f = mutex.async_lock().and_then(
      [&](std::unique_lock<mellon::future_mutex<default_test_tag>>&& lock) noexcept -> std::monostate {
        executed = true;
        EXPECT_TRUE(mutex.is_locked());
        return std::monostate();
      });

  ASSERT_FALSE(executed);
  mutex.unlock();  // now the future should execute by the executor

  // this should eventually be resolved
  std::move(f).await(mellon::yes_i_know_that_this_call_will_block);
  ASSERT_TRUE(executed);
  ASSERT_FALSE(mutex.is_locked());
}

TEST(locks, async_lock_test2) {
  mellon::future_mutex<default_test_tag> mutex;
  mutex.lock();

  ASSERT_TRUE(mutex.is_locked());

  std::atomic<int> executed = 0;
  // this should not execute immediately
  auto f = mutex.async_lock().and_then(
      [&](std::unique_lock<mellon::future_mutex<default_test_tag>>&& lock) noexcept -> std::monostate {
        EXPECT_TRUE(mutex.is_locked());
        EXPECT_EQ(executed, 0);
        executed = 1;
        return std::monostate();
      });

  auto f2 = mutex.async_lock().and_then(
      [&](std::unique_lock<mellon::future_mutex<default_test_tag>>&& lock) noexcept -> std::monostate {
        EXPECT_TRUE(mutex.is_locked());
        EXPECT_EQ(executed, 1);
        executed = 2;
        return std::monostate();
      });

  ASSERT_FALSE(executed);
  mutex.unlock();  // now the future should execute by the executor

  // this should eventually be resolved
  std::move(f).await(mellon::yes_i_know_that_this_call_will_block);
  std::move(f2).await(mellon::yes_i_know_that_this_call_will_block);
  ASSERT_EQ(executed, 2);
  ASSERT_FALSE(mutex.is_locked());
}
