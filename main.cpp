#include <cassert>
#include <iostream>
#include <thread>

#include <futures/completion-queue.h>
#include <futures/futures.h>
#include <futures/utilities.h>

struct constructor_counter_base {
  constructor_counter_base(constructor_counter_base const&) = delete;
  constructor_counter_base(constructor_counter_base&&) noexcept
      : _memory(new int(4)) {
    std::cout << "new counter is " << ++_counter << std::endl;
  }
  constructor_counter_base() noexcept : _memory(new int(4)) {
    std::cout << "new counter is " << ++_counter << std::endl;
  }
  ~constructor_counter_base() {
    auto x = _counter.fetch_sub(1);
    std::cout << "new counter is " << (x - 1) << std::endl;
    if (x == 0) {
      std::abort();
    }
    delete _memory;
  }

  int* _memory;
  static std::atomic<std::size_t> _counter;
};

std::atomic<std::size_t> constructor_counter_base::_counter = 0;

template <typename T>
struct constructor_counter : constructor_counter_base {
  static std::atomic<std::size_t> _counter;

  ~constructor_counter() = default;
  template <typename... Args>
  explicit constructor_counter(std::in_place_t, Args&&... args)
      : _value(std::forward<Args>(args)...) {}
  constructor_counter(T v) : _value(std::move(v)) {}
  constructor_counter(constructor_counter const& v) = default;
  constructor_counter(constructor_counter&& v) noexcept = default;
  constructor_counter& operator=(constructor_counter const& v) = default;
  constructor_counter& operator=(constructor_counter&& v) noexcept = default;

  T& operator*() { return _value; }
  T const& operator*() const { return _value; }
  T* operator->() { return &_value; }
  T const* operator->() const { return &_value; }

 private:
  T _value;
};

using namespace futures;

auto baz() -> future<constructor_counter<int>> {
  auto&& [f, p] = futures::make_promise<constructor_counter<int>>();

  /* */
  std::thread t([p = std::move(p)]() mutable {
    std::this_thread::sleep_for(2 * std::chrono::seconds{1});
    std::cout << "thread done" << std::endl;
    std::move(p).fulfill(std::in_place, 12);
  });
  t.detach();

  return std::move(f);
}

auto foo() -> future<expect::expected<constructor_counter<int>>> {
  return baz()
      .and_capture([](constructor_counter<int>&& x) { return std::move(x); })
      .then([](constructor_counter<int>&& x) {
        std::cout << "first then executed " << *x << std::endl;
        return *x + 4;
      })
      .then([](constructor_counter<int>&& x) -> constructor_counter<int> {
        std::cout << "second then executed " << *x << std::endl;
        // throw std::runtime_error("foobar");
        return 1;
      });
}

auto bar() -> future<int> { return future<int>(std::in_place, 12); }

template<typename Duration>
auto bar(Duration d) -> future<int> {
  auto&& [f, p] = make_promise<int>();

  std::thread([p = std::move(p), d]() mutable {
    std::this_thread::sleep_for(d);
    std::cout << "thread done" << std::endl;
    std::move(p).fulfill(5);
  }).detach();

  return std::move(f);
}

int main() {
  /*
  auto value =
      foo()
          .then([](constructor_counter<int>&& x) noexcept {
            std::cout << "third then executed " << *x << std::endl;
            return *x + 4;
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


  std::vector<future<constructor_counter<int>>> v;
  v.emplace_back(foo().unwrap_or(4));
  v.emplace_back(bar().as<constructor_counter<int>>());
  v.emplace_back(foo().unwrap_or(12));

  // TODO make this preserve the order
  auto w = futures::collect(v.begin(),
  v.end()).await(futures::yes_i_know_that_this_call_will_block); for (auto&& x :
  w) { std::cout << *x << std::endl;
  }*/

  auto f = future<int>(std::in_place, 12).and_then([](int x) noexcept { return x * 2; });

  future<double> fd = std::move(f).as<double>();

/*
  auto queue = std::make_shared<completion_context<future<int>>>();
  queue->register_future(bar(std::chrono::seconds{1}));

  auto t = std::thread([&] {
    while (auto value = queue->await()) {
      std::cout << "completed future " << *value << std::endl;
    }
  });

  for (size_t i = 0; i < 1000; i++) {
    queue->register_future(bar(std::chrono::seconds{4}));
    queue->register_future(bar(std::chrono::seconds{1}));
    queue->register_future(bar(std::chrono::seconds{10}));
  }

  t.join();
  std::cout << constructor_counter_base::_counter << std::endl;
  */
}
