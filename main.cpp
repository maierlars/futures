#include <iostream>
#include <thread>

#include "futures/futures.h"

using namespace futures;

struct A {
  explicit A(int x) : x(std::make_shared<int>(x)) {}
  std::shared_ptr<int> x = nullptr;
};

auto baz() -> future<A> {
  auto&& [f, p] = make_promise<A>();

/*
  std::thread t([p = std::move(p)]() mutable {
    std::this_thread::sleep_for(2 * std::chrono::milliseconds{1});
    std::cout << "thread done" << std::endl;
    std::move(p).fulfill(12);
  });
  t.detach();*/
  std::move(p).fulfill(12);

  return std::move(f);
}

auto foo() -> future<int> {
  return baz().and_then_temp([](A&& x) noexcept { return A(*x.x + 4); }).and_then([](A&& x) noexcept {
    return *x.x - 4;
  });
}

int main() {
  foo().finally([](int x) noexcept {
    std::cout << x << std::endl;
  });

  /*foo()
      .and_then_temp([](int x) noexcept { return x * x + 3; })
      .and_then([](int x) noexcept { return x / 4; })
      .abandon();*/
  std::this_thread::sleep_for(std::chrono::seconds{2});
}
