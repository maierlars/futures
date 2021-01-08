#ifndef FUTURES_TRAITS_H
#define FUTURES_TRAITS_H

namespace mellon {

template <typename, typename>
struct future;
template <typename, typename, typename, typename>
struct future_temporary;
struct default_tag;

template <typename F>
struct future_trait;

template <typename T>
struct tag_trait;

template <typename T, typename Tag>
struct future_trait<future<T, Tag>> {
  static constexpr auto is_value_inlined = future<T, Tag>::is_value_inlined;

  template <typename U>
  using future_for_type = future<U, Tag>;
  using value_type = T;
  using tag_type = Tag;
};

template <typename T, typename F, typename R, typename Tag>
struct future_trait<future_temporary<T, F, R, Tag>> : future_trait<future<R, Tag>> {};

template <typename F>
using future_value_type_t = typename future_trait<F>::value_type;
template <typename F>
using future_tag_type_t = typename future_trait<F>::tag_type;
template <typename F>
inline constexpr auto is_future_value_inlined = future_trait<F>::is_value_inlined;

namespace detail {
template <typename Tag>
struct tag_trait_helper {
  template <typename tag, typename = void>
  struct has_assertion_handler : std::false_type {};
  template <typename tag>
  struct has_assertion_handler<tag, std::void_t<typename tag_trait<tag>::assertion_handler>>
      : std::true_type {};

  template <typename tag, typename T, typename = void>
  struct has_promise_aborted_handler : std::false_type {};
  template <typename tag, typename T>
  struct has_promise_aborted_handler<tag, T, std::void_t<typename tag_trait<tag>::template promise_abandoned_handler<T>>>
      : std::true_type {};

  template <typename tag, typename T, typename = void>
  struct has_future_aborted_handler : std::false_type {};
  template <typename tag, typename T>
  struct has_future_aborted_handler<tag, T, std::void_t<typename tag_trait<tag>::template future_abandoned_handler<T>>>
      : std::true_type {};

  template <typename tag, typename = void>
  struct has_small_value_size : std::false_type {};
  template <typename tag>
  struct has_small_value_size<tag, std::void_t<decltype(tag_trait<tag>::small_value_size)>>
      : std::true_type {};

  static void assert_true(bool condition) noexcept {
    if constexpr (has_assertion_handler<Tag>::value) {
      using assertion_handler = typename tag_trait<Tag>::assertion_handler;
      static_assert(std::is_nothrow_invocable_r_v<void, assertion_handler, bool>);
      assertion_handler{}(condition);
    } else {
      tag_trait_helper<default_tag>::assert_true(condition);
    }
  }

  template <typename T>
  static T abandon_promise() noexcept {
    if constexpr(has_promise_aborted_handler<Tag, T>::value) {
    using handler = typename tag_trait<Tag>::template abandoned_promise_handler<T>;
    return handler{}();
    } else {
      return tag_trait_helper<default_tag>::abandon_promise<T>();
    }
  }

  template <typename T, typename U>
  static void abandon_future(U&& u) noexcept {
    if constexpr (has_promise_aborted_handler<Tag, T>::value) {
      using handler = typename tag_trait<Tag>::template abandoned_promise_handler<T>;
      handler{}(std::forward<U>(u));
    } else {
      tag_trait_helper<default_tag>::abandon_future<T>(std::forward<U>(u));
    }
  }

  static constexpr std::size_t small_value_size() {
    if constexpr (has_small_value_size<Tag>::value) {
      return tag_trait<Tag>::small_value_size;
    } else {
      return tag_trait_helper<default_tag>::small_value_size();
    }
  }
};
}  // namespace detail
}  // namespace mellon

#endif  // FUTURES_TRAITS_H
