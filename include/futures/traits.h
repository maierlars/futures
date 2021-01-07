#ifndef FUTURES_TRAITS_H
#define FUTURES_TRAITS_H

namespace futures {

template <typename>
struct future;

template <typename F>
struct future_traits;

template <typename T>
struct future_traits<future<T>> {
  static constexpr auto is_value_inlined = future<T>::is_value_inlined;

  template <typename U>
  using future_for_type = future<U>;
  using value_type = T;
};

template <typename F>
using future_value_type_t = typename future_traits<F>::value_type;
template <typename F>
inline constexpr auto is_future_value_inlined = future_traits<F>::is_value_inlined;

}  // namespace futures

#endif  // FUTURES_TRAITS_H
