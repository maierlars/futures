#include <iostream>
#include <thread>

#include "futures/futures.h"
#include "futures/utilities.h"

template <typename T>
using my_future = futures::future<expect::expected<T>>;

struct A {
  explicit A(int x) noexcept : x(std::make_shared<int>(x)) {}
  std::shared_ptr<int> x = nullptr;
};

void foobar() {
  auto&& [f, p] = futures::make_promise<int>();

  // mutex.lock();
  std::move(f).and_then([](int x) noexcept {
    // mutex.unlock();
    return 14;
  });

  throw 12;
}

auto baz() -> my_future<A> {
  auto&& [f, p] = futures::make_promise<expect::expected<A>>();

  /* */
  std::thread t([p = std::move(p)]() mutable {
    std::this_thread::sleep_for(2 * std::chrono::seconds{1});
    std::cout << "thread done" << std::endl;
    std::move(p).fulfill(std::in_place, 12);
  });
  t.detach();

  return std::move(f);
}

auto foo() -> my_future<int> {
  return baz()
      .then([](A&& x) {
        std::cout << "first then executed " << *x.x << std::endl;
        return A(*x.x + 4);
      })
      .then([](A&& x) {
        std::cout << "second then executed " << *x.x << std::endl;
        // throw std::runtime_error("foobar");

        std::cout << "not returning a value " << *x.x << std::endl;
        return *x.x - 4;
      });
}

auto bar() -> my_future<int> { return my_future<int>(std::in_place, 12); }

int main() {
  auto value = foo()
               .then([](int&& x) noexcept {
                 std::cout << "third then executed " << x << std::endl;
                 return x + 4;
               })
               .rethrow_nested<std::logic_error>("runtime error is not allowed")
               .catch_error<std::runtime_error>([](auto&& e) noexcept -> int {
                 std::cout << "caught runtime error " << e.what() << std::endl;
                 return 5;
               })
               .await_unwrap();
  std::cout << "awaited value " << value << std::endl;

  auto&& [a, b] = futures::collect(foo(), baz()).transpose();
  std::cout << "collect returned" << std::endl;

  futures::collect(std::move(a), std::move(b)).get<0>().await(futures::yes_i_know_that_this_call_will_block);

  std::cout << "second collect returned" << std::endl;

  std::vector<my_future<int>> v;
  v.emplace_back(foo());
  v.emplace_back(bar());
  v.emplace_back(foo());

  // TODO make this preserve the order
  auto w = futures::collect(v.begin(), v.end()).await(futures::yes_i_know_that_this_call_will_block);
  for (auto&& x : w) {
    std::cout << x.unwrap() << std::endl;
  }

  std::this_thread::sleep_for(std::chrono::seconds{5});
}
