#ifndef FUTURES_SEQUENCER_H
#define FUTURES_SEQUENCER_H
#include "futures.h"
#include "utilities.h"

namespace mellon {



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
  void run_next(std::tuple<Ts...>&& t) noexcept {
    if constexpr (I == sizeof...(Is)) {
      std::move(promise).fulfill_from_tuple(t);
    } else {
      // invoke the function
      auto result = invoke_nth<I>(std::move(t));
      using result_type = decltype(result);
      static_assert(is_future_v<result_type>);
      using future_result = typename result_type::value_type;
      std::move(result).finally([self = this->shared_from_this()](future_result&& tuple) noexcept {
        self->template run_next<I + 1>(std::make_tuple(tuple));
      });
    }
  }

 private:
  template <std::size_t I, typename F = std::tuple_element_t<I, std::tuple<Fs...>>>
  auto nth_function() -> index_function_tag<I, F>& {
    return *this;
  }

  template <std::size_t I, typename... Ts>
  auto invoke_nth(std::tuple<Ts...>&& t) {
    return std::apply(nth_function<I>(), std::move(t));
  }

  mellon::promise<R, Tag> promise;
};

template <typename...>
struct sequence_builder_impl;
template <typename Tag, typename InputType, typename OutputType, typename... Fs>
using sequence_builder =
    sequence_builder_impl<Tag, InputType, OutputType, std::index_sequence_for<Fs...>, Fs...>;

template <typename FutureTag, typename InputType, typename OutputType, std::size_t... Is, typename... Fs>
struct sequence_builder_impl<FutureTag, InputType, OutputType, std::index_sequence<Is...>, Fs...>
    : private index_function_tag<Is, Fs>... {
  static_assert(is_tuple_v<OutputType>);

  template <typename G, typename R = std::invoke_result_t<G, OutputType>>
  auto append(G&& g) {
    static_assert(is_future_v<R>);
    using new_output_type = typename R::value_type;
    static_assert(is_tuple_v<new_output_type>);
    return sequence_builder<FutureTag, InputType, new_output_type, Fs..., std::decay_t<G>>(
        std::in_place, std::move(nth_function<Is>())..., std::forward<G>(g));
  }

  template <typename G, std::enable_if_t<is_applicable_v<std::is_invocable, G, OutputType>, int> = 0,
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
  }

  template <typename... Gs>
  explicit sequence_builder_impl(std::in_place_t, Gs&&... gs)
      : index_function_tag<Is, Fs>(std::forward<Gs>(gs))... {}

  // template <typename U = OutputType, std::enable_if_t<std::tuple_size_v<U> != 1, int> = 0>
  auto compose() -> mellon::future<OutputType, FutureTag> {
    auto&& [f, p] = mellon::make_promise<OutputType, FutureTag>();
    auto machine =
        std::make_shared<sequence_state_machine<FutureTag, OutputType, Fs...>>(
            std::move(nth_function<Is>())..., std::move(p));
    machine->template run_next<0>(std::make_tuple());
    return std::move(f);
  }

  /*template <typename U = OutputType, std::enable_if_t<std::tuple_size_v<U> == 1, int> = 0,
            typename T = std::tuple_element_t<0, U>>
  auto compose() -> mellon::future<T, FutureTag> {
    auto&& [f, p] = mellon::make_promise<OutputType, FutureTag>();
    auto machine =
        std::make_shared<sequence_state_machine<FutureTag, OutputType, Fs...>>(
            std::move(p), std::move(nth_function<Is>())...);
    return std::move(f).template get<0>();
  }*/

 private:
  template <typename... Ts>
  static auto collect(std::tuple<Ts...>&& ts) {
    return mellon::collect(std::move(ts));
  }

  template <std::size_t I, typename F = std::tuple_element_t<I, std::tuple<Fs...>>>
  auto& nth_function() {
    return index_function_tag<I, F>::ref();
  }
};

template <typename T, typename Tag>
struct init_sequence {
  explicit init_sequence(future<T, Tag>&& f) : init_future(std::move(f)) {}

  auto operator()() noexcept -> future<T, Tag> {
    return std::move(init_future);
  }

  future<T, Tag> init_future;
};

template <typename T, typename FutureTag>
auto sequence(future<T, FutureTag> f) {
  return sequence_builder<FutureTag, std::tuple<>, std::tuple<T>, init_sequence<T, FutureTag>>(
      std::in_place, init_sequence(std::move(f)));
}

}  // namespace mellon

#endif  // FUTURES_SEQUENCER_H
