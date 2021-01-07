#ifndef FUTURES_TEST_HELPER_H
#define FUTURES_TEST_HELPER_H
#include <futures/futures.h>

#include <memory>
#include <thread>

template <typename F>
struct delayed_context : F {
  explicit delayed_context(F f) : F(std::move(f)) {}

  void run() {
    F::operator()();
  }

  void trigger() {
    run();
  }
};

template <typename F>
struct delayed_guard {
  explicit delayed_guard(std::shared_ptr<delayed_context<F>> ctx)
      : ctx(std::move(ctx)) {}
  delayed_guard(delayed_guard const&) = delete;
  delayed_guard(delayed_guard&&) noexcept = default;
  delayed_guard& operator=(delayed_guard const&) = delete;
  delayed_guard& operator=(delayed_guard&&) noexcept = default;

  ~delayed_guard() = default;

  void trigger() && {
    ctx->trigger();
  }

 private:
  std::shared_ptr<delayed_context<F>> ctx;
};

template <typename F, typename... Args>
auto trigger_delayed(F&& f) noexcept {
  auto ctx = std::make_shared<delayed_context<F>>(std::forward<F>(f));
  return delayed_guard{ctx};
}

template <typename T, typename... Args>
auto fulfill_delayed(futures::promise<T, futures::default_tag>&& p, Args&&... args) noexcept {
  return trigger_delayed(
      [p = std::move(p), args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::move(p).fulfill_from_tuple(args_tuple);
      });
}

template <typename T, typename... Args>
auto make_delayed_fulfilled(Args&&... args) {
  auto&& [f, p] = futures::make_promise<T>();
  return std::make_pair(std::move(f),
                        fulfill_delayed(std::move(p), std::forward<Args>(args)...));
}

template <typename T>
auto abandon_delayed(futures::promise<T, futures::default_tag>&& p) noexcept {
  return trigger_delayed([p = std::move(p)]() mutable { std::move(p).abandon(); });
}

template <typename T>
auto make_delayed_abandon() {
  auto&& [f, p] = futures::make_promise<T>();
  return std::make_pair(std::move(f), abandon_delayed(std::move(p)));
}

struct constructor_counter_base {
  constructor_counter_base(constructor_counter_base const&) = delete;
  constructor_counter_base(constructor_counter_base&&) noexcept
      : _memory(new int(4)) {}
  constructor_counter_base() noexcept : _memory(new int(4)) {}
  ~constructor_counter_base() {
    auto x = _counter.fetch_sub(1);
    if (x == 0) {
      std::abort();
    }
    delete _memory;
  }

  int* _memory;
  static std::atomic<std::size_t> _counter;
};

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

#endif  // FUTURES_TEST_HELPER_H
