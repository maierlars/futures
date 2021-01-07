#ifndef FUTURES_TRAITS_H
#define FUTURES_TRAITS_H

namespace futures {

template <typename, typename>
struct future;

template <typename F>
struct future_traits;

template <typename T, typename Tag>
struct future_traits<future<T, Tag>> {
  static constexpr auto is_value_inlined = future<T, Tag>::is_value_inlined;

  template <typename U>
  using future_for_type = future<U, Tag>;
  using value_type = T;
  using tag_type = Tag;
};

template <typename T, typename F, typename R, typename Tag>
struct future_traits<future_temporary<T, F, R, Tag>> : future_traits<future<R, Tag>> {};

template <typename F>
using future_value_type_t = typename future_traits<F>::value_type;
template <typename F>
using future_tag_type_t = typename future_traits<F>::tag_type;
template <typename F>
inline constexpr auto is_future_value_inlined = future_traits<F>::is_value_inlined;

}  // namespace futures

#endif  // FUTURES_TRAITS_H
