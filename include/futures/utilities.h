#ifndef FUTURES_UTILITIES_H
#define FUTURES_UTILITIES_H
#include <atomic>
#include <bitset>
#include <tuple>
#include "futures.h"

namespace futures {

namespace detail {
template <typename...>
struct collect_context_impl;

template <typename... Ts, std::size_t... Is>
struct collect_context_impl<std::index_sequence<Is...>, Ts...> : detail::box<Ts, Is>... {
  using tuple_type = std::tuple<Ts...>;
  static constexpr auto fan_in_degree = sizeof...(Ts);

  ~collect_context_impl() noexcept {
    if (valid_values.all()) {
      std::move(out_promise).fulfill(detail::box<Ts, Is>::cast_move()...);
    } else {
      std::move(out_promise).abandon();
    }

    (std::invoke([this] {
       if (valid_values.test(Is)) {
         static_assert(std::is_nothrow_destructible_v<Ts>);
         detail::box<Ts, Is>::destroy();
       }
     }),
     ...);
  }

  template <std::size_t I, typename... Args, typename T = std::tuple_element_t<I, tuple_type>>
  void fulfill(Args&&... args) noexcept {
    static_assert(I < fan_in_degree);
    static_assert(std::is_nothrow_constructible_v<T, Args...>);
    std::unique_lock guard(mutex);
    detail::box<std::tuple_element_t<I, tuple_type>, I>::emplace(std::forward<Args>(args)...);
    valid_values.set(I);
  }

  explicit collect_context_impl(promise<tuple_type> p)
      : out_promise(std::move(p)) {}

  // TODO this is not as good as it could be.
  // We have two allocations, one for the context and one for the promise.
  // Furthermore, we should get rid ot the mutex.
  // Third, we use a shared pointer that has more overhead than necessary.
 private:
  promise<tuple_type> out_promise;
  std::bitset<fan_in_degree> valid_values;
  std::mutex mutex;
};

template <typename... Ts>
using collect_context = collect_context_impl<std::index_sequence_for<Ts...>, Ts...>;

template <typename... Ts, std::size_t... Is, typename R = std::tuple<Ts...>>
auto collect(std::index_sequence<Is...>, future<Ts>&&... fs) -> future<R> {
  auto&& [f, p] = make_promise<R>();
  auto ctx = std::make_shared<detail::collect_context<Ts...>>(std::move(p));

  (std::invoke(
       [&] { std::move(fs).finally([ctx](Ts&& t) noexcept { ctx->template fulfill<Is>(std::move(t)); }); }),
   ...);

  return std::move(f);
}

}  // namespace detail

template <typename... Ts, typename R = std::tuple<Ts...>>
auto collect(future<Ts>&&... fs) -> future<R> {
  return detail::collect(std::index_sequence_for<Ts...>{}, std::move(fs)...);
}



}  // namespace futures

#endif  // FUTURES_UTILITIES_H
