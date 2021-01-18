#ifndef FUTURES_SEQUENCER_H
#define FUTURES_SEQUENCER_H
#include "futures.h"
#include "utilities.h"

namespace mellon {

template <typename T>
inline constexpr auto always_false_v = false;

template <typename Tag, typename... Ts>
struct multi_resources : std::tuple<future<Ts, Tag>...> {
  explicit multi_resources(std::tuple<future<Ts, Tag>...> t)
      : std::tuple<future<Ts, Tag>...>(std::move(t)) {}
};
template <typename T>
struct is_multi_resources : std::false_type {};
template <typename Tag, typename... Ts>
struct is_multi_resources<multi_resources<Tag, Ts...>> : std::true_type {};
template <typename T>
inline constexpr auto is_multi_resources_v = is_multi_resources<T>::value;

template <typename Tag, typename... Ts>
auto mr(Ts&&... ts) {
  return multi_resources(std::make_tuple([&] {
    using T = std::decay_t<Ts>;
    if constexpr (is_future_v<T>) {
      return std::forward<Ts>(ts);
    } else {
      return future<T, Tag>{std::in_place, std::forward<Ts>(ts)};
    }
  }()...));
}

template <std::size_t, typename F>
struct index_function_tag : F {
  explicit index_function_tag(F f) : F(std::move(f)) {}
  F& ref() { return *this; }
};

template <typename...>
struct sequence_state_machine_impl;

template <typename Tag, typename R, typename... Fs>
using sequence_state_machine =
    sequence_state_machine_impl<Tag, R, std::index_sequence_for<Fs...>, Fs...>;

template <typename Tag, typename R, std::size_t... Is, typename... Fs>
struct sequence_state_machine_impl<Tag, R, std::index_sequence<Is...>, Fs...>
    : private index_function_tag<Is, Fs>...,
      std::enable_shared_from_this<sequence_state_machine_impl<Tag, R, std::index_sequence<Is...>, Fs...>> {
  explicit sequence_state_machine_impl(Fs... fs, promise<R, Tag> promise)
      : index_function_tag<Is, Fs>(std::move(fs))..., promise(std::move(promise)) {}

  template <std::size_t I, typename... Ts>
  void run_next(Ts&&... t) noexcept {
    if constexpr (I == sizeof...(Is)) {
      // last state fulfills the promise
      std::move(promise).fulfill(std::forward<Ts>(t)...);
    } else {
      // invoke the function
      auto future = invoke_nth<I>(std::forward<Ts>(t)...);
      using future_type = decltype(future);
      static_assert(is_future_v<future_type>);
      using future_result = typename future_type::value_type;
      std::move(future).finally([self = this->shared_from_this()](future_result&& result) noexcept {
        self->template run_next<I + 1>(std::move(result));
      });
    }
  }

 private:
  template <std::size_t I, typename F = std::tuple_element_t<I, std::tuple<Fs...>>>
  auto nth_function() noexcept -> index_function_tag<I, F>& {
    return *this;
  }

  template <std::size_t I, typename... Ts>
  auto invoke_nth(Ts&&... t) noexcept {
    static_assert(std::is_nothrow_invocable_v<nth_function_type<I>, Ts&&...>);
    return std::invoke(nth_function<I>(), std::forward<Ts>(t)...);
  }

  template <std::size_t I>
  using nth_function_type = std::tuple_element_t<I, std::tuple<Fs...>>;

  mellon::promise<R, Tag> promise;
};

template <typename T, typename FutureTag>
auto sequence(future<T, FutureTag> f);

struct empty_init_sequence_start_t {};
inline constexpr auto empty_init_sequence_start = empty_init_sequence_start_t{};

template <typename...>
struct sequence_builder_impl;
template <typename Tag, typename InputType, typename OutputType, typename... Fs>
using sequence_builder =
    sequence_builder_impl<Tag, InputType, OutputType, std::index_sequence_for<Fs...>, Fs...>;

template <typename FutureTag, typename InputType, typename OutputType, std::size_t... Is, typename... Fs>
struct sequence_builder_impl<FutureTag, InputType, OutputType, std::index_sequence<Is...>, Fs...>
    : private index_function_tag<Is, Fs>... {
  template <typename G, std::enable_if_t<std::is_nothrow_invocable_v<G, OutputType&&>, int> = 0,
            typename ReturnValue = std::invoke_result_t<G, OutputType&&>,
            std::enable_if_t<is_future_v<ReturnValue>, int> = 0, typename ValueType = typename ReturnValue::value_type>
  auto append(G&& g) && /* TODO exception specifier */ {
    return sequence_builder<FutureTag, InputType, ValueType, Fs..., std::decay_t<G>>(
        std::in_place, std::move(nth_function<Is>())..., std::forward<G>(g));
  }

  /*template <typename G, std::enable_if_t<is_applicable_v<std::is_invocable, G, OutputType>, int> = 0,
            typename R = apply_result_t<G, OutputType>>
  auto then_do(G&& g) {
    if constexpr (is_future_v<R>) {
      return append([g = std::forward<G>(g)](OutputType params) mutable noexcept {
        return std::apply(g, std::move(params));
      });
    } else if constexpr (is_multi_resources_v<R>) {
      return append([g = std::forward<G>(g)](OutputType params) mutable noexcept {
        return collect(std::apply(g, params));
      });
    } else {
      return append([g = std::forward<G>(g)](OutputType params) mutable noexcept {
        return future<R, FutureTag>{std::in_place, std::apply(g, params)};
      });
    }
  }*/

  auto compose() && -> mellon::future<OutputType, FutureTag> {
    /* TODO exception specifier -- do we need a nothrow alloc? */
    auto&& [f, p] = mellon::make_promise<OutputType, FutureTag>();
    auto machine =
        std::make_shared<sequence_state_machine<FutureTag, OutputType, Fs...>>(
            std::move(nth_function<Is>())..., std::move(p));
    machine->template run_next<0>(empty_init_sequence_start);
    return std::move(f);
  }

  template <typename G, std::enable_if_t<std::is_invocable_v<G, OutputType&&>, int> = 0,
            typename ReturnValue = std::invoke_result_t<G, OutputType&&>,
            std::enable_if_t<is_future_v<ReturnValue>, int> = 0, typename ValueType = typename ReturnValue::value_type>
  auto append_capture(G&& g) && /* TODO exception specifier */ {
    return move_self().append([g = std::forward<G>(g)](OutputType&& t) mutable noexcept
                              -> future<expect::expected<ValueType>, FutureTag> {
      // expected<future<T>>
      auto result = expect::captured_invoke(g, std::move(t));
      if (result.has_value()) {
        // if T == expected<U>
        if constexpr (expect::is_expected_v<ValueType>) {
          return std::move(result.unwrap());
        } else {
          // the result has to become a expected
          return std::move(result).unwrap().template as<expect::expected<ValueType>>();
        }
      } else {
        if constexpr (expect::is_expected_v<ValueType>) {
          return future<expect::expected<typename ValueType::value_type>, FutureTag>{
              std::in_place, result.error()};
        } else {
          return future<expect::expected<ValueType>, FutureTag>{std::in_place,
                                                                result.error()};
        }
      }
    });
  }

  template <typename G, typename U = OutputType, std::enable_if_t<expect::is_expected_v<U>, int> = 0,
            typename V = typename U::value_type, typename ReturnValue = std::invoke_result_t<G, V&&>,
            std::enable_if_t<is_future_v<ReturnValue>, int> = 0, typename ValueType = typename ReturnValue::value_type,
            std::enable_if_t<!expect::is_expected_v<ValueType>, int> = 0>
  auto then_do(G&& g) && {
    return move_self().append_capture(
        [g = std::forward<G>(g)](OutputType&& v) mutable -> future<ValueType, FutureTag> {
          return std::invoke(g, std::move(v).unwrap());
        });
  }

  template <typename G, typename U = OutputType, std::enable_if_t<expect::is_expected_v<U>, int> = 0,
            typename V = typename U::value_type, std::enable_if_t<is_tuple_v<V>, int> = 0,
            typename ReturnValue = apply_result_t<G, V>,
            std::enable_if_t<is_future_v<ReturnValue>, int> = 0, typename ValueType = typename ReturnValue::value_type,
            std::enable_if_t<!expect::is_expected_v<ValueType>, int> = 0>
  auto then_do(G&& g) && {
    return move_self().append_capture(
        [g = std::forward<G>(g)](OutputType&& v) mutable -> future<ValueType, FutureTag> {
          return std::apply(g, std::move(v).unwrap());
        });
  }

 private:
  template <typename S, typename TT>
  friend auto sequence(future<S, TT> f);
  template <typename...>
  friend struct sequence_builder_impl;

  template <typename... Gs>
  explicit sequence_builder_impl(std::in_place_t, Gs&&... gs)
      : index_function_tag<Is, Fs>(std::forward<Gs>(gs))... {}

  template <typename... Ts>
  static auto collect(std::tuple<Ts...>&& ts) {
    return mellon::collect(std::move(ts));
  }

  template <std::size_t I, typename F = std::tuple_element_t<I, std::tuple<Fs...>>>
  auto& nth_function() {
    return index_function_tag<I, F>::ref();
  }

  auto&& move_self() { return std::move(*this); }
};

template <typename T, typename Tag>
struct init_sequence {
  explicit init_sequence(future<T, Tag>&& f) : init_future(std::move(f)) {}

  auto operator()(empty_init_sequence_start_t) noexcept -> future<T, Tag> {
    return std::move(init_future);
  }

  future<T, Tag> init_future;
};

template <typename T, typename FutureTag>
auto sequence(future<T, FutureTag> f) {
  return sequence_builder<FutureTag, empty_init_sequence_start_t, T, init_sequence<T, FutureTag>>(
      std::in_place, init_sequence(std::move(f)));
}

}  // namespace mellon

#endif  // FUTURES_SEQUENCER_H
