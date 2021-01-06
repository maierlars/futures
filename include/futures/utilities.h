#ifndef FUTURES_UTILITIES_H
#define FUTURES_UTILITIES_H
#include <atomic>
#include <bitset>
#include <tuple>
#include "futures.h"

namespace futures {
namespace detail {
template <typename...>
struct collect_tuple_context_impl;

template <typename... Ts, std::size_t... Is>
struct collect_tuple_context_impl<std::index_sequence<Is...>, Ts...>
    : detail::box<Ts, Is>... {
  using tuple_type = std::tuple<Ts...>;
  static constexpr auto fan_in_degree = sizeof...(Ts);

  ~collect_tuple_context_impl() noexcept {
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

  explicit collect_tuple_context_impl(promise<tuple_type> p)
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
using collect_tuple_context =
    collect_tuple_context_impl<std::index_sequence_for<Ts...>, Ts...>;

template <typename... Ts, std::size_t... Is, typename R = std::tuple<Ts...>>
auto collect(std::index_sequence<Is...>, future<Ts>&&... fs) -> future<R> {
  auto&& [f, p] = make_promise<R>();
  auto ctx = std::make_shared<detail::collect_tuple_context<Ts...>>(std::move(p));

  (std::invoke([&] {
     std::move(fs).finally(
         [ctx](Ts&& t) noexcept { ctx->template fulfill<Is>(std::move(t)); });
   }),
   ...);

  return std::move(f);
}

}  // namespace detail

/**
 * Collect creates a new my_future that awaits all given futures and completes when
 * all futures have received their value. It then returns a tuple containing all
 * the individual values.
 *
 * If a promise of the collected futures is abandoned, the new promise will be
 * abandoned as well.
 * @tparam Ts value types of the futures (deduced)
 * @param fs Futures that are collected.
 * @return A new my_future that returns a tuple.
 */
template <typename... Ts, typename R = std::tuple<Ts...>>
auto collect(future<Ts>&&... fs) -> future<R> {
  return detail::collect(std::index_sequence_for<Ts...>{}, std::move(fs)...);
}

/**
 * Collects all futures in the range [begin, end). The result is new my_future
 * returning a `std::vector<T>`. The results might appear in a different order
 * then they appeared in the input range. If a promise is abandoned, its result
 * is omitted from the result vector.
 * @tparam InputIt
 * @param begin
 * @param end
 * @return
 */
template <typename InputIt, typename V = typename std::iterator_traits<InputIt>::value_type,
          std::enable_if_t<is_future_v<V>, int> = 0,
          typename B = typename V::value_type, typename R = std::vector<B>>
auto collect(InputIt begin, InputIt end) -> future<R> {
  auto&& [f, p] = make_promise<R>();

  // TODO this does two allocations
  struct context {
    ~context() { std::move(out_promise).fulfill(std::move(result)); }
    explicit context(promise<std::vector<B>> p) : out_promise(std::move(p)) {}
    promise<std::vector<B>> out_promise;
    std::vector<B> result;
  };

  auto ctx = std::make_shared<context>(std::move(p));
  // FIXME: do we actually need `distance` here?
  // We have to do a reserve here, otherwise the emplace_back might throw an
  // exception but is has to be noexcept.
  ctx->result.reserve(std::distance(begin, end));
  for (std::size_t i = 0; begin != end; ++begin, ++i) {
    std::move(*begin).finally([i, ctx](B&& v) noexcept {
      // never throws because we have reserved enough memory
      ctx->result[i] = std::move(v);
    });
  }
  return std::move(f);
}

}  // namespace futures

#endif  // FUTURES_UTILITIES_H
