#ifndef FUTURES_FUTURES_H
#define FUTURES_FUTURES_H
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include "expected.h"
#include "traits.h"

namespace mellon {

namespace detail {

template <typename T>
struct no_deleter {
  void operator()(T*) const noexcept {}
};

template <typename T>
using unique_but_not_deleting_pointer = std::unique_ptr<T, no_deleter<T>>;  // TODO find a better name

struct invalid_pointer_type {};

extern invalid_pointer_type invalid_pointer_inline_value;
extern invalid_pointer_type invalid_pointer_future_abandoned;
extern invalid_pointer_type invalid_pointer_promise_abandoned;
extern invalid_pointer_type invalid_pointer_promise_fulfilled;

#define FUTURES_INVALID_POINTER_INLINE_VALUE(T) \
  reinterpret_cast<::mellon::detail::continuation_base<T>*>(&detail::invalid_pointer_inline_value)
#define FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T) \
  reinterpret_cast<::mellon::detail::continuation<T>*>(&detail::invalid_pointer_future_abandoned)
#define FUTURES_INVALID_POINTER_PROMISE_ABANDONED(T) \
  reinterpret_cast<::mellon::detail::continuation<T>*>(&detail::invalid_pointer_promise_abandoned)
#define FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T) \
  reinterpret_cast<::mellon::detail::continuation<T>*>(&detail::invalid_pointer_promise_fulfilled)

template <typename T, std::size_t tag = 0>
struct box {
  static_assert(!std::is_reference_v<T>);
  box() noexcept = default;

  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  explicit box(std::in_place_t,
               Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    emplace(std::forward<Args>(args)...);
  }

  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  void emplace(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    new (ptr()) T(std::forward<Args>(args)...);
  }

  void destroy() noexcept(std::is_nothrow_destructible_v<T>) {
    std::destroy_at(ptr());
  }

  T* ptr() { return reinterpret_cast<T*>(value.data()); }
  T const* ptr() const { return reinterpret_cast<T const*>(value.data()); }

  T& ref() & { return *ptr(); }
  T const& ref() const& { return *ptr(); }
  T&& ref() && { return std::move(*ptr()); }

  T& operator*() & { return ref(); }
  T const& operator*() const& { return ref(); }
  T&& operator*() && { return std::move(ref()); }

  T* operator->() { return ptr(); }
  T const* operator->() const { return ptr(); }

  template <typename U = T, std::enable_if_t<std::is_move_constructible_v<U>, int> = 0>
  T&& cast_move() {
    return std::move(ref());
  }
  template <typename U = T, std::enable_if_t<!std::is_move_constructible_v<U>, int> = 0>
  T& cast_move() {
    return ref();
  }

 private:
  alignas(T) std::array<std::byte, sizeof(T)> value;
};

template <>
struct box<void> {
  box() noexcept = default;
  explicit box(std::in_place_t) noexcept {}
  void emplace() noexcept {}
};

template <typename T, bool inline_value>
struct small_box : box<void> {
  static constexpr bool stores_value = false;
};
template <typename T>
struct small_box<T, true> : box<T> {
  using box<T>::box;
  static constexpr bool stores_value = true;
};

}  // namespace detail

struct promise_abandoned_error : std::exception {
  [[nodiscard]] const char* what() const noexcept override;
};

struct default_tag {};

template <>
struct tag_trait<default_tag> {
  struct assertion_handler {
    void operator()(bool test) const noexcept {
      if (!test) {
        std::abort();
      }
    }
  };

  template <typename T>
  struct abandoned_future_handler {
    void operator()(T&&) noexcept {}  // this is like ignoring a return value
  };

  template <typename T>
  struct abandoned_future_handler<expect::expected<T>> {
    void operator()(expect::expected<T>&& e) noexcept {
      if (e.has_error()) {
        e.rethrow_error();  // this is like an uncaught exception
      }
    }
  };

  template <typename T>
  struct abandoned_promise_handler {
    T operator()() noexcept { std::abort(); }  // in general we can not continue
  };

  template <typename T>
  struct abandoned_promise_handler<expect::expected<T>> {
    expect::expected<T> operator()() noexcept {
      // at least in this case we can throw a nice exception :)
      return std::make_exception_ptr(promise_abandoned_error{});
    }
  };

  static constexpr auto small_value_size = 64;
};

template <typename T, typename Tag>
struct future;
template <typename T, typename Tag>
struct promise;
template <typename T>
struct is_future : std::false_type {};
template <typename T, typename Tag>
struct is_future<future<T, Tag>> : std::true_type {};
template <typename T>
inline constexpr auto is_future_v = is_future<T>::value;
template <typename T>
struct is_future_temporary : std::false_type {};
template <typename T, typename F, typename R, typename Tag>
struct is_future_temporary<future_temporary<T, F, R, Tag>> : std::true_type {};
template <typename T>
inline constexpr auto is_future_temporary_v = is_future_temporary<T>::value;
template <typename T>
struct is_future_like : std::false_type {};
template <typename T, typename Tag>
struct is_future_like<future<T, Tag>> : std::true_type {};
template <typename T, typename F, typename R, typename Tag>
struct is_future_like<future_temporary<T, F, R, Tag>> : std::true_type {};
template <typename T>
inline constexpr auto is_future_like_v = is_future_like<T>::value;

namespace detail {

template <typename Tag, typename T>
struct handler_helper {
  static T abandon_promise() noexcept {
    return detail::tag_trait_helper<Tag>::template abandon_promise<T>();
  }

  template <typename U>
  static void abandon_future(U&& u) noexcept {
    detail::tag_trait_helper<Tag>::template abandon_future<T>(std::forward<U>(u));
  }
};

template <typename T>
struct continuation;
template <typename T>
struct continuation_base;
template <typename T>
struct continuation_start;
template <typename T, typename F, typename R>
struct continuation_step;
template <typename T, typename F>
struct continuation_final;

template <typename T, typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
void fulfill_continuation(continuation_base<T>* base,
                          Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
  base->emplace(std::forward<Args>(args)...);  // this can throw an exception

  // the remainder should be noexcept
  static_assert(std::is_nothrow_destructible_v<T>,
                "type should be nothrow destructible.");
  std::invoke([&]() noexcept {
    continuation<T>* expected = nullptr;
    if (!base->_next.compare_exchange_strong(expected, FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T),
                                             std::memory_order_release,
                                             std::memory_order_acquire)) {
      if (expected != FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T)) {
        std::invoke(*expected, base->cast_move());
        // FIXME this is a recursive call. Build this a loop!
      }

      base->destroy();
      delete base;
    }
  });
}

template <typename Tag, typename T>
void abandon_continuation(continuation_base<T>* base) noexcept {
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T),
                                           std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    if (expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
      detail::handler_helper<Tag, T>::abandon_future(base->cast_move());
      static_assert(std::is_nothrow_destructible_v<T>);
      base->destroy();
    } else {
      detail::tag_trait_helper<Tag>::assert_true(
          expected == FUTURES_INVALID_POINTER_PROMISE_ABANDONED(T));
    }

    delete base;
  }
}

template <typename Tag, typename T>
void abandon_promise(continuation_start<T>* base) noexcept {
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, FUTURES_INVALID_POINTER_PROMISE_ABANDONED(T),
                                           std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    if (expected == FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T)) {
      delete base;  // we all agreed on not having this promise-init_future-chain
    } else {
      return fulfill_continuation(base, detail::handler_helper<Tag, T>::abandon_promise());
    }
  }
}

template <typename Tag, typename T, typename... Args>
auto allocate_frame_noexcept(Args&&... args) noexcept -> T* {
  static_assert(std::is_nothrow_constructible_v<T, Args...>,
                "type should be nothrow constructable");
  auto frame = new (std::nothrow) T(std::forward<Args>(args)...);
  detail::tag_trait_helper<Tag>::assert_true(frame != nullptr);
  return frame;
}

template <typename Tag, typename T, typename F>
void insert_continuation_final(continuation_base<T>* base, F&& f) noexcept {
  static_assert(std::is_nothrow_invocable_r_v<void, F, T&&>);
  static_assert(std::is_nothrow_destructible_v<T>);
  if (base->_next.load(std::memory_order_acquire) ==
      FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
    // short path
    std::invoke(std::forward<F>(f), base->cast_move());
    base->destroy();
    delete base;
    return;
  }

  auto step =
      detail::allocate_frame_noexcept<Tag, continuation_final<T, F>>(std::in_place,
                                                                     std::forward<F>(f));
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, step, std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    detail::tag_trait_helper<Tag>::assert_true(
        expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T));
    std::invoke(step->function_self(), base->cast_move());
    base->destroy();
    delete base;
    delete step;
  }
}

template <typename Tag, typename T, typename F, typename R, typename G>
auto insert_continuation_step(continuation_base<T>* base, G&& f) noexcept
    -> future<R, Tag> {
  static_assert(std::is_nothrow_invocable_r_v<R, F, T&&>);
  static_assert(std::is_nothrow_destructible_v<T>);

  if (base->_next.load(std::memory_order_acquire) ==
      FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
    // short path
    static_assert(std::is_nothrow_move_constructible_v<R>);
    auto fut = future<R, Tag>{std::in_place,
                              std::invoke(std::forward<G>(f), base->cast_move())};
    base->destroy();
    delete base;
    return fut;
  }

  auto step = detail::allocate_frame_noexcept<Tag, continuation_step<T, F, R>>(
      std::in_place, std::forward<G>(f));
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, step, std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    detail::tag_trait_helper<Tag>::assert_true(
        expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T));
    if constexpr (future<R, Tag>::is_value_inlined) {
      auto fut = future<R, Tag>{std::in_place, std::invoke(step->function_self(),
                                                           base->cast_move())};
      base->destroy();
      delete base;
      delete step;
      return fut;
    } else {
      step->emplace(std::invoke(step->function_self(), base->cast_move()));
      base->destroy();
      delete base;
    }
  }

  return future<R, Tag>{step};
}

template <typename T>
struct continuation {
  virtual ~continuation() = default;
  virtual void operator()(T&&) noexcept = 0;
};

template <typename T>
struct continuation_base : box<T> {
  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  explicit continuation_base(std::in_place_t, Args&&... args) noexcept(
      std::is_nothrow_constructible_v<box<T>, std::in_place_t, Args...>)
      : box<T>(std::in_place, std::forward<Args>(args)...),
        _next(FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {}

  template <typename U = T, std::enable_if_t<std::is_void_v<U>, int> = 0>
  explicit continuation_base(std::in_place_t) noexcept
      : box<T>(std::in_place), _next(FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {}

  continuation_base() noexcept : box<T>() {}
  virtual ~continuation_base() = default;

  std::atomic<continuation<T>*> _next = nullptr;
};

template <typename T>
struct continuation_start final : continuation_base<T> {};

template <typename F, typename Func = std::decay_t<F>, typename = std::enable_if_t<std::is_class_v<Func>>>
struct function_store : Func {
  template <typename G = F>
  explicit function_store(std::in_place_t,
                          G&& f) noexcept(std::is_nothrow_constructible_v<Func, G>)
      : Func(std::forward<G>(f)) {}

  [[nodiscard]] Func& function_self() { return *this; }
  [[nodiscard]] Func const& function_self() const { return *this; }
};

template <typename T, typename F, typename R>
struct continuation_step final : continuation_base<R>, function_store<F>, continuation<T> {
  static_assert(std::is_nothrow_invocable_r_v<R, F, T&&>);

  template <typename G = F>
  explicit continuation_step(std::in_place_t, G&& f) noexcept(
      std::is_nothrow_constructible_v<function_store<F>, std::in_place_t, G>)
      : function_store<F>(std::in_place, std::forward<G>(f)) {}

  void operator()(T&& t) noexcept override {
    detail::fulfill_continuation(this, std::invoke(function_store<F>::function_self(),
                                                   std::move(t)));
  }
};

template <typename T, typename F>
struct continuation_final final : continuation<T>, function_store<F> {
  static_assert(std::is_nothrow_invocable_r_v<void, F, T>);
  template <typename G = F>
  explicit continuation_final(std::in_place_t, G&& f) noexcept(
      std::is_nothrow_constructible_v<function_store<F>, std::in_place_t, G>)
      : function_store<F>(std::in_place, std::forward<G>(f)) {}
  void operator()(T&& t) noexcept override {
    std::invoke(function_store<F>::function_self(), std::move(t));
    delete this;
  }
};

template <typename F, int>
struct composer_tag : function_store<F> {
  using function_store<F>::function_store;
  using function_store<F>::function_self;
};

template <typename F, typename G>
struct composer : composer_tag<F, 0>, composer_tag<G, 1> {
  template <typename S, typename T>
  explicit composer(S&& s, T&& t)
      : composer_tag<F, 0>(std::in_place, std::forward<S>(s)),
        composer_tag<G, 1>(std::in_place, std::forward<T>(t)) {}

  composer(composer&&) noexcept = default;

  template <typename... Args>
  auto operator()(Args&&... args) noexcept {
    static_assert(std::is_nothrow_invocable_v<G, Args...>);
    using return_value = std::invoke_result_t<G, Args...>;
    static_assert(std::is_nothrow_invocable_v<F, return_value>);

    return std::invoke(composer_tag<F, 0>::function_self(),
                       std::invoke(composer_tag<G, 1>::function_self(),
                                   std::forward<Args>(args)...));
  }
};

template <typename F, typename G>
auto compose(F&& f, G&& g) {
  return composer<F, G>(std::forward<F>(f), std::forward<G>(g));
}

template <template <typename...> typename T, typename...>
struct unpack_tuple_into;

template <template <typename...> typename T, typename... Vs, typename... Us>
struct unpack_tuple_into<T, std::tuple<Us...>, Vs...> : T<Vs..., Us...> {};

template <template <typename...> typename T, typename... Vs>
inline constexpr auto unpack_tuple_into_v = unpack_tuple_into<T, Vs...>::value;

}  // namespace detail

template <typename T, typename Tag>
struct promise_type_based_extension {};

/**
 * Producing end of init_future-chain. When the promise is `fulfilled` the chain
 * of mellon is evaluated.
 * @tparam T
 */
template <typename T, typename Tag>
struct promise : promise_type_based_extension<T, Tag>,
                 user_defined_promise_additions_t<Tag, T> {
  /**
   * Destroies the promise. If the promise has not been fulfilled or moved away
   * it will be abandoned.
   */
  ~promise() {
    if (_base) {
      std::move(*this).abandon();
    }
  }

  promise(promise const&) = delete;
  promise& operator=(promise const&) = delete;
  promise(promise&& o) noexcept = default;
  promise& operator=(promise&& o) noexcept = default;

  /**
   * In place constructs the result using the given parameters.
   * @tparam Args
   * @param args
   */
  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  void fulfill(Args&&... args) && noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    detail::fulfill_continuation(_base.get(), std::forward<Args>(args)...);
    _base.reset();  // we can not use _base.release() because the constructor of T could
    // throw an exception. In that case the promise has to stay active.
  }

  /**
   * Abandons the promise. The `abandoned_promise_handler` will be called.
   * This either generates a default value for `T` or calls `std::abort()`.
   */
  void abandon() && { detail::abandon_promise<Tag>(_base.release()); }

  template <typename Tuple, typename tuple_type = std::remove_reference_t<Tuple>>
  void fulfill_from_tuple(Tuple&& t) && noexcept(
      detail::unpack_tuple_into_v<std::is_nothrow_constructible, tuple_type, T>) {
    return std::move(*this).fulfill_from_tuple_impl(
        std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
  }

  [[nodiscard]] bool empty() const noexcept { return _base == nullptr; }

 private:
  template <typename Tuple, std::size_t... Is, typename tuple_type = std::remove_reference_t<Tuple>>
  void fulfill_from_tuple_impl(Tuple&& t, std::index_sequence<Is...>) && noexcept(
      std::is_nothrow_constructible_v<T, std::tuple_element_t<Is, tuple_type>...>) {
    static_assert(std::is_constructible_v<T, std::tuple_element_t<Is, tuple_type>...>);
    std::move(*this).fulfill(std::get<Is>(std::forward<Tuple>(t))...);
  }

  template <typename S, typename STag>
  friend auto make_promise() -> std::pair<future<S, STag>, promise<S, STag>>;
  explicit promise(detail::continuation_start<T>* base) : _base(base) {}
  detail::unique_but_not_deleting_pointer<detail::continuation_start<T>> _base = nullptr;
};

template <typename T, typename F, typename R, typename Tag>
struct future_temporary;

namespace detail {
template <typename T, typename F, typename Tag>
struct future_temporary_proxy {
  template <typename R>
  using instance = future_temporary<T, F, R, Tag>;
};
template <typename Tag>
struct future_proxy {
  template <typename R>
  using instance = future<R, Tag>;
};
}  // namespace detail

struct yes_i_know_that_this_call_will_block_t {};
inline constexpr yes_i_know_that_this_call_will_block_t yes_i_know_that_this_call_will_block;

template <typename T, template <typename> typename F, typename Tag>
struct future_type_based_extensions;
namespace detail {


template<typename Tag, typename T, template <typename> typename F, typename Ts, typename = void>
struct has_constructor : std::false_type {};
template<typename Tag, typename T, template <typename> typename F, typename... Ts>
struct has_constructor<Tag, T, F, std::tuple<Ts...>, std::void_t<decltype(future_type_based_extensions<T, F, Tag>::construct(std::declval<Ts>()...))>> : std::true_type {};
template<typename Tag, typename T, template <typename> typename F, typename... Ts>
inline constexpr auto has_constructor_v = has_constructor<Tag, T, F, std::tuple<Ts...>>::value;


/**
 * Base class unifies all common operations on mellon and future_temporaries.
 * Futures and Temporaries are dervived from this class and only specialise
 * the functions `and_then` and `finally`.
 * @tparam T Value type
 * @tparam Fut Parent class template expecting one parameter.
 */
template <typename Tag, typename T, template <typename> typename Fut>
struct future_base_base {
  static_assert(!std::is_void_v<T>);

  using value_type = T;

  /**
   * _Blocks_ the current thread until the init_future is fulfilled. This **not**
   * something you should do unless you have a very good reason to do so. The
   * whole point of mellon is to make code non-blocking.
   *
   * @return Returns the result value of type T.
   */
  T await(yes_i_know_that_this_call_will_block_t) && {
    detail::box<T> box;
    bool is_waiting = false, has_value = false;
    std::mutex mutex;
    std::condition_variable cv;
    std::move(self()).finally([&](T&& v) noexcept {
      bool was_waiting;
      {
        std::unique_lock guard(mutex);
        box.template emplace(std::move(v));
        has_value = true;
        was_waiting = is_waiting;
      }
      if (was_waiting) {
        cv.notify_one();
      }
    });
    std::unique_lock guard(mutex);
    is_waiting = true;
    cv.wait(guard, [&] { return has_value; });
    T value(std::move(box).ref());
    box.destroy();
    return value;
  }

  /**
   * Calls `f` and captures its return value in an `expected<R>`.
   * @tparam F
   * @param f
   * @return
   */
  template <typename F, std::enable_if_t<std::is_invocable_v<F, T&&>, int> = 0,
            typename R = std::invoke_result_t<F, T&&>>
  auto and_capture(F&& f) && noexcept {
    return std::move(self()).and_then([f = std::forward<F>(f)](T&& v) noexcept {
      return expect::captured_invoke(f, std::move(v));
    });
  }

  /**
   * Converts the content to `U`.
   * @tparam U
   * @return
   */
  template <typename U, std::enable_if_t<std::is_convertible_v<T, U>, int> = 0>
  auto as() && {
    static_assert(!expect::is_expected_v<T>);
    if constexpr (std::is_same_v<T, U>) {
      return std::move(self());
    } else {
      return std::move(self()).and_then(
          [](T&& v) noexcept -> U { return std::move(v); });
    }
  }

  void fulfill(promise<T, Tag>&& p) && noexcept {
    std::move(self()).finally([p = std::move(p)](T && t) mutable noexcept {
      std::move(p).fulfill(std::move(t));
    });
  }

 private:
  auto& self() noexcept { return *static_cast<Fut<T>*>(this); }
  auto& self() const noexcept { return *static_cast<Fut<T> const*>(this); }
};

template <typename T, template <typename> typename Fut, typename Tag>
struct future_base : user_defined_additions_t<Tag, T, Fut>,
                     future_type_based_extensions<T, Fut, Tag> {
  static_assert(std::is_base_of_v<future_base_base<Tag, T, Fut>, future_type_based_extensions<T, Fut, Tag>>);
};

template <typename Tag, typename T>
using small_box_tag =
    small_box<T, tag_trait_helper<Tag>::template is_type_inlined<T>()>;
}  // namespace detail

template <typename T, template <typename> typename F, typename Tag>
struct future_type_based_extensions : detail::future_base_base<Tag, T, F> {};

/**
 * A temporary object that is used to chain together multiple `and_then` calls
 * into a single step to reduce memory and allocation overhead. Is derived from
 * future_base and thus provides all functions that mellon do.
 *
 * The temporary is implicitly convertible to `init_future<T>`.
 * @tparam T Initial value type.
 * @tparam F Temporary function type. Chain of functions that have been applied.
 * @tparam R Result type of the chain.
 */
template <typename T, typename F, typename R, typename Tag>
struct future_temporary
    : detail::future_base<R, detail::future_temporary_proxy<T, F, Tag>::template instance, Tag>,
      private detail::function_store<F>,
      private detail::small_box_tag<Tag, T> {
  static constexpr auto is_value_inlined = detail::small_box_tag<Tag, T>::stores_value;

  static_assert(!is_future_v<R>,
                "init_future<init_future<T>> is a bad idea and thus not supported");

  future_temporary(future_temporary const&) = delete;
  future_temporary& operator=(future_temporary const&) = delete;
  future_temporary(future_temporary&& o) noexcept
      : detail::function_store<F>(std::move(o)), _base(nullptr) {
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::small_box_tag<Tag, T>::emplace(o.cast_move());
        o.destroy();
      }
    }
    std::swap(_base, o._base);
  }
  future_temporary& operator=(future_temporary&& o) noexcept {
    if (_base) {
      std::move(*this).abandon();
    }
    detail::tag_trait_helper<Tag>::assert_true(_base == nullptr);
    detail::function_store<F>::operator=(std::move(o));
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::small_box_tag<Tag, T>::emplace(o.cast_move());
        o.destroy();
      }
    }
    std::swap(_base, o._base);
  }

  ~future_temporary() {
    if (_base) {
      std::move(*this).abandon();
    }
  }

  template <typename G, std::enable_if_t<std::is_nothrow_invocable_v<G, R&&>, int> = 0,
            typename S = std::invoke_result_t<G, R&&>>
  auto and_then(G&& f) && noexcept {
    auto&& composition =
        detail::compose(std::forward<G>(f),
                        std::move(detail::function_store<F>::function_self()));

    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        auto fut = future_temporary<T, decltype(composition), S, Tag>(
            std::in_place, std::move(composition),
            detail::small_box_tag<Tag, T>::cast_move());
        cleanup_local_state();
        return fut;
      }
    }

    return future_temporary<T, decltype(composition), S, Tag>(std::move(composition),
                                                              std::move(_base));
  }

  template <typename G, std::enable_if_t<std::is_nothrow_invocable_r_v<void, G, R&&>, int> = 0>
  void finally(G&& f) && noexcept {
    auto&& composition =
        detail::compose(std::forward<G>(f),
                        std::move(detail::function_store<F>::function_self()));

    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        std::invoke(composition, detail::small_box_tag<Tag, T>::cast_move());
        cleanup_local_state();
        return;
      }
    }

    return detail::insert_continuation_final<Tag, T, decltype(composition)>(
        _base.release(), std::move(composition));
  }

  void abandon() && noexcept { std::move(*this).finalize().abandon(); }

  /* implicit */ operator future<R, Tag>() && noexcept {
    return std::move(*this).finalize();
  }

  auto finalize() && noexcept -> future<R, Tag> {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        static_assert(std::is_nothrow_move_constructible_v<R>);
        auto f =
            future<R, Tag>(std::in_place,
                           std::invoke(detail::function_store<F>::function_self(),
                                       detail::small_box_tag<Tag, T>::cast_move()));
        static_assert(std::is_nothrow_destructible_v<T>);
        cleanup_local_state();
        return f;
      }
    }

    return detail::insert_continuation_step<Tag, T, F, R>(
        _base.release(), std::move(detail::function_store<F>::function_self()));
  }

  // TODO move to future_base
  [[nodiscard]] bool holds_inline_value() const noexcept {
    return is_value_inlined && _base.get() == FUTURES_INVALID_POINTER_INLINE_VALUE(T);
  }
  [[nodiscard]] bool empty() const noexcept { return _base == nullptr; }

 private:
  void cleanup_local_state() {
    detail::small_box_tag<Tag, T>::destroy();
    _base.reset();
  }
  template <typename G = F>
  future_temporary(G&& f, detail::unique_but_not_deleting_pointer<detail::continuation_base<T>> base)
      : detail::function_store<F>(std::in_place, std::forward<G>(f)),
        _base(std::move(base)) {}
  template <typename G = F, typename S = T>
  future_temporary(std::in_place_t, G&& f, S&& s)
      : detail::function_store<F>(std::in_place, std::forward<G>(f)),
        detail::small_box_tag<Tag, T>(std::in_place, std::forward<S>(s)),
        _base(FUTURES_INVALID_POINTER_INLINE_VALUE(T)) {}

  template <typename, typename>
  friend struct future;
  template <typename, typename, typename, typename>
  friend struct future_temporary;
  // TODO move _base pointer to future_base.
  detail::unique_but_not_deleting_pointer<detail::continuation_base<T>> _base;
};


/**
 * Consuming end of a init_future-chain. You can add more elements to the chain using
 * this interface.
 * @tparam T value_type
 */
template <typename T, typename Tag>
struct future
    : detail::future_base<T, detail::future_proxy<Tag>::template instance, Tag>,
      private detail::small_box_tag<Tag, T> {
  /**
   * Is true if the init_future can store an inline value.
   */
  static constexpr auto is_value_inlined = detail::small_box_tag<Tag, T>::stores_value;

  static_assert(!std::is_void_v<T>,
                "void is not supported, use std::monostate instead");
  static_assert(!is_future_v<T>,
                "init_future<init_future<T>> is a bad idea and thus not supported");
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_destructible_v<T>);

  future(future const&) = delete;
  future& operator=(future const&) = delete;
  future(future&& o) noexcept : _base(nullptr) {
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::small_box_tag<Tag, T>::emplace(o.cast_move());
        o.destroy();
      }
    }
    std::swap(_base, o._base);
  }
  future& operator=(future&& o) noexcept {
    if (_base) {
      std::move(*this).abandon();
    }
    detail::tag_trait_helper<Tag>::assert_true(_base == nullptr);
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::small_box_tag<Tag, T>::emplace(o.cast_move());
        o.destroy();  // o will have _base == nullptr
      }
    }
    std::swap(_base, o._base);
    return *this;
  }

  template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  future(future<U, Tag>&& o) noexcept : future(std::move(o).template as<T>()) {}
  template <typename U, typename F, typename S, std::enable_if_t<std::is_convertible_v<U, T>, int> = 0>
  future(future_temporary<S, F, U, Tag>&& o) noexcept
      : future(std::move(o).template as<U>().finalize()) {}

  /**
   * If the init_future was not used or moved away, the init_future is abandoned.
   * For more, see `abandon`.
   */
  ~future() {
    if (_base) {
      std::move(*this).abandon();
    }
  }

  /**
   * Constructs a fulfilled init_future in place. The arguments are passed to the
   * constructor of `T`. If the value can be inlined it is constructed inline,
   * otherwise memory is allocated.
   *
   * This constructor is noexcept if the used constructor of `T`. If the value
   * is not small, `std::bad_alloc` can occur.
   * @tparam Args
   * @param args
   */
  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  explicit future(std::in_place_t, Args&&... args) noexcept(
      std::conjunction_v<std::is_nothrow_constructible<T, Args...>, std::bool_constant<is_value_inlined>>) {
    if constexpr (is_value_inlined) {
      detail::small_box_tag<Tag, T>::emplace(std::forward<Args>(args)...);
      _base.reset(FUTURES_INVALID_POINTER_INLINE_VALUE(T));
    } else {
      _base.reset(new detail::continuation_base<T>(std::in_place,
                                                   std::forward<Args>(args)...));
    }
  }

  explicit future(T t) : future(std::in_place, std::move(t)) {}

  template <typename... Ts, typename U = T,
      std::enable_if_t<std::is_constructible_v<U, Ts...>, int> = 0,
            std::enable_if_t<detail::has_constructor_v<Tag, T, detail::future_proxy<Tag>::template instance, Ts...>, int> = 0>
  explicit future(Ts&&... ts)
      : future(future_type_based_extensions<T, detail::future_proxy<Tag>::template instance, Tag>::construct(
            std::forward<Ts>(ts)...)) {}

  /**
   * (fmap) Enqueues a callback to the init_future chain and returns a new init_future that awaits
   * the return value of the provided callback. It is undefined in which thread
   * the callback is executed. `F` must be nothrow invocable with `T&&` as parameter.
   *
   * This function returns a temporary object which is implicitly convertible to init_future<R>.
   *
   * @tparam F
   * @param f
   * @return A new init_future with `value_type` equal to the result type of `F`.
   */
  template <typename F, std::enable_if_t<std::is_nothrow_invocable_v<F, T&&>, int> = 0,
            typename R = std::invoke_result_t<F, T&&>>
  auto and_then(F&& f) && noexcept {
    if constexpr (detail::tag_trait_helper<Tag>::is_disable_temporaries()) {
      return std::move(*this).and_then_direct(std::forward<F>(f));
    } else {
      if constexpr (is_value_inlined) {
        if (holds_inline_value()) {
          auto fut =
              future_temporary<T, F, R, Tag>(std::in_place, std::forward<F>(f),
                                             detail::small_box_tag<Tag, T>::cast_move());
          cleanup_local_state();
          return fut;
        }
      }
      return future_temporary<T, F, R, Tag>(std::forward<F>(f), std::move(_base));
    }
  }

  /**
   * (fmap) Enqueues a callback to the init_future chain and returns a new init_future that awaits
   * the return value of the provided callback. It is undefined in which thread
   * the callback is executed. `F` must be nothrow invocable with `T&&` as parameter.
   *
   * Unlike `and_then` this function does not return a temporary.
   *
   * @tparam F callback type
   * @param f callback
   * @return A new init_future with `value_type` equal to the result type of `F`.
   */
  template <typename F, std::enable_if_t<std::is_nothrow_invocable_v<F, T&&>, int> = 0,
            typename R = std::invoke_result_t<F, T&&>>
  auto and_then_direct(F&& f) && noexcept -> future<R, Tag> {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        auto fut = future<R, Tag>{std::in_place,
                                  std::invoke(std::forward<F>(f), this->cast_move())};
        cleanup_local_state();
        return std::move(fut);
      }
    }
    return detail::insert_continuation_step<Tag, T, F, R>(_base.release(),
                                                          std::forward<F>(f));
  }

  /**
   * Enqueues a final callback and ends the init_future chain.
   *
   * @tparam F callback type
   * @param f callback
   */
  template <typename F, std::enable_if_t<std::is_nothrow_invocable_r_v<void, F, T&&>, int> = 0>
  void finally(F&& f) && noexcept {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        std::invoke(std::forward<F>(f), detail::small_box_tag<Tag, T>::cast_move());
        cleanup_local_state();
        return;
      }
    }

    return detail::insert_continuation_final<Tag, T, F>(_base.release(),
                                                        std::forward<F>(f));
  }

  /**
   * Returns true if the init_future holds a value locally.
   * @return true if a local value is present.
   */
  [[nodiscard]] bool holds_inline_value() const noexcept {
    return is_value_inlined && _base.get() == FUTURES_INVALID_POINTER_INLINE_VALUE(T);
  }
  // TODO move _base pointer to future_base.
  [[nodiscard]] bool empty() const noexcept { return _base == nullptr; }

  /**
   * Abandons a init_future chain. If the promise is abandoned as well, nothing happens.
   * If, however, the promise is fulfilled it depends on the `abandoned_future_handler`
   * for that type what will happen next. In most cases this is just cleanup.
   *
   * Some types, e.g. `expected<S>` will call terminated if it contains an
   * unhandled exception.
   */
  void abandon() && noexcept {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        cleanup_local_state();
        return;
      }
    }

    detail::abandon_continuation<Tag>(_base.release());
  }

  explicit future(detail::continuation_base<T>* ptr) noexcept : _base(ptr) {}

 private:
  template <typename, typename>
  friend struct future;

  void cleanup_local_state() {
    detail::small_box_tag<Tag, T>::destroy();
    _base.reset();
  }
  // TODO move _base pointer to future_base.
  detail::unique_but_not_deleting_pointer<detail::continuation_base<T>> _base;
};

template <typename T, template <typename> typename Fut, typename Tag>
struct future_type_based_extensions<expect::expected<T>, Fut, Tag>
    : detail::future_base_base<Tag, expect::expected<T>, Fut> {
  /**
   * If the `expected<T>` contains a value, the callback is called with the value.
   * Otherwise it is not executed. Any thrown exception is captured by the `expected<T>`.
   * @tparam F
   * @tparam R
   * @param f
   * @return
   */
  template <typename F, typename U = T, std::enable_if_t<std::is_invocable_v<F, U&&>, int> = 0,
            typename R = std::invoke_result_t<F, U&&>>
  auto then(F&& f) && noexcept {
    // TODO what if `F` returns an `expected<U>`. Do we want to flatten automagically?
    static_assert(std::is_nothrow_move_constructible_v<F>);
    return std::move(self()).and_then(
        [f = std::forward<F>(f)](expect::expected<T>&& e) mutable noexcept
        -> expect::expected<R> { return std::move(e).map_value(f); });
  }

  template <typename F, typename U = T, std::enable_if_t<std::is_void_v<U>, int> = 0,
            std::enable_if_t<std::is_invocable_v<F>, int> = 0, typename R = std::invoke_result_t<F>>
  auto then(F&& f) && noexcept {
    return std::move(self()).and_then(
        [f = std::forward<F>(f)](expect::expected<T>&& e) mutable noexcept
        -> expect::expected<R> { return std::move(e).map_value(f); });
  }

  /**
   * If the `expected<T>` contains a value, the callback is called with the value.
   * Otherwise it is not executed. Any thrown exception is captured by the `expected<T>`.
   * @tparam F
   * @tparam R
   * @param f
   * @return
   */
  /*template <typename F, std::enable_if_t<std::is_invocable_v<F, expect::expected<T>&&>, int> = 0,
            typename R = std::invoke_result_t<F, expect::expected<T>&&>,
            typename U = T, std::enable_if_t<!expect::is_expected_v<U>, int> = 0>
  auto then(F&& f) && noexcept {
    // TODO what if `F` returns an `expected<U>`. Do we want to flatten automagically?
    return std::move(self()).and_capture(std::forward<F>(f));
  }*/

  /**
   * Catches an exception of type `E` and calls `f` with `E const&`. If the
   * values does not contain an exception `f` will _never_ be called. The return
   * value of `f` is replaced with the exception. If `f` itself throws an
   * exception, the old exception is replaced with the new one.
   * @tparam E
   * @tparam F
   * @param f
   * @return A new init_future containing a value if the exception was caught.
   */
  template <typename E, typename F, typename U = T,
      std::enable_if_t<std::is_invocable_r_v<U, F, E const&>, int> = 0>
  auto catch_error(F&& f) && noexcept {
    return std::move(self()).and_then(
        [f = std::forward<F>(f)](expect::expected<T>&& e) noexcept -> expect::expected<T> {
          return std::move(e).template map_error<E>(f);
        });
  }

  /**
   * Same as await but unwraps the `expected<T>`.
   * @return Returns the value contained in expected, or throws the
   * contained exception.
   */
  template <typename U = T, std::enable_if_t<!std::is_void_v<U>, int> = 0>
  T await_unwrap() {
    return std::move(self()).await(yes_i_know_that_this_call_will_block).unwrap();
  }

  template <typename U = T, std::enable_if_t<std::is_void_v<U>, int> = 0>
  void await_unwrap() {
    std::move(self()).await(yes_i_know_that_this_call_will_block);
  }

  /**
   * Extracts the value from the `expected<T>` or replaces any exception with
   * a `T` constructed using the given arguments.
   *
   * Note that those arguments are copied into the chain and only used,
   * when required. If they are not needed, they are discarded.
   * @tparam Args
   * @param args
   * @return
   */
  template <typename... Args>
  auto unwrap_or(Args&&... args) {
    return std::move(self()).and_then(
        [args_tuple = std::make_tuple(std::forward<Args>(args)...)](
            expect::expected<T>&& e) noexcept {
          if (e.has_value()) {
            return std::move(e).unwrap();
          } else {
            return std::make_from_tuple<T>(std::move(args_tuple));
          }
        });
  }

  /**
   * (join) Flattens the underlying `expected<T>`, i.e. converts
   * `expected<expected<T>>` to `expected<T>`.
   * @return Future containing just a `expected<T>`.
   */
  template <typename U = T, std::enable_if_t<expect::is_expected_v<U>, int> = 0>
  auto flatten() {
    return std::move(self()).and_then(
        [](expect::expected<T>&& e) noexcept { return std::move(e).flatten(); });
  }

  /**
   * Rethrows any exception as a nested exception combined with `E(Args...)`.
   * `E` is only constructed if an exception was found. The arguments are copied
   * into the chain.
   * @tparam E
   * @tparam Args
   * @param args
   * @return
   */
  template <typename E, typename... Args>
  auto rethrow_nested(Args&&... args) noexcept {
    return std::move(self()).and_then(
        [args_tuple = std::make_tuple(std::forward<Args>(args)...)](
            expect::expected<T>&& e) mutable noexcept -> expect::expected<T> {
          // TODO can we instead forward to expected<T>::rethrow_nested
          try {
            try {
              e.rethrow_error();
            } catch (...) {
              std::throw_with_nested(std::make_from_tuple<E>(std::move(args_tuple)));
            }
          } catch (...) {
            return std::current_exception();
          }

          return std::move(e);
        });
  }

  /**
   * Rethrows an exception of type `W` as a nested exception combined with
   * `E(Args...)`. `E` is only constructed if an exception was found. The
   * arguments are copied into the chain.
   * @tparam E
   * @tparam Args
   * @param args
   * @return
   */
  template <typename W, typename E, typename... Args>
  auto rethrow_nested_if(Args&&... args) noexcept {
    return std::move(self()).and_then(
        [args_tuple = std::make_tuple(std::forward<Args>(args)...)](
            expect::expected<T>&& e) mutable noexcept -> expect::expected<T> {
          // TODO can we instead forward to expected<T>::rethrow_nested
          try {
            try {
              e.rethrow_error();
            } catch (W const&) {
              std::throw_with_nested(std::make_from_tuple<E>(std::move(args_tuple)));
            }
          } catch (...) {
            return std::current_exception();
          }
          return std::move(e);
        });
  }

  /**
   * Converts an `expected<T>` to `expected<U>`.
   * @tparam U
   * @return
   */
  template <typename U, std::enable_if_t<expect::is_expected_v<U>, int> = 0, typename R = typename U::value_type>
  auto as() {
    return std::move(self()).and_then(
        [](expect::expected<T>&& e) noexcept -> expect::expected<R> {
          return std::move(e).template as<R>();
        });
  }


 private:
  using future_type = Fut<expect::expected<T>>;

  future_type& self() noexcept { return *static_cast<future_type*>(this); }
  future_type const& self() const noexcept {
    return *static_cast<future_type const*>(this);
  }

 public:
  template<typename... Ts, std::enable_if_t<std::is_constructible_v<T, Ts...>, int> = 0>
  static auto construct(std::in_place_t, Ts&&... ts) -> future_type {
    return future_type{std::in_place, std::in_place, std::forward<Ts>(ts)...};
  }
};

static_assert(detail::has_constructor_v<default_tag, expect::expected<int>, detail::future_proxy<default_tag>::template instance, std::in_place_t, int>);

template <typename T, typename Tag>
struct promise_type_based_extension<expect::expected<T>, Tag> {
  /**
   * A special form of `fulfill`. Uses `capture_invoke` to call a function and
   * capture the return value of any exceptions in the promise.
   * @tparam F
   * @tparam Args
   * @param f
   * @param args
   */
  template <typename F, typename... Args, std::enable_if_t<std::is_invocable_r_v<T, F, Args...>, int> = 0>
  void capture(F&& f, Args&&... args) && noexcept {
    std::move(self()).fulfill(
        expect::captured_invoke(std::forward<F>(f), std::forward<Args>(args)...));
  }

  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  void fulfill(Args&&... args) && noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    std::move(self()).fulfill(std::in_place, std::forward<Args>(args)...);
  }

  /**
   * Throws `e` into the promise.
   * @tparam E
   * @param e
   */
  template <typename E>
  void throw_into(E&& e) noexcept {
    std::move(self()).fulfill(std::make_exception_ptr(std::forward<E>(e)));
  }

  /**
   * Constructs an exception of type `E` with the given arguments
   * and passes it into the promise.
   * @tparam E
   * @param e
   */
  template <typename E, typename... Args, std::enable_if_t<std::is_constructible_v<E, Args...>, int> = 0>
  void throw_exception(Args&&... args) noexcept(std::is_nothrow_constructible_v<E, Args...>) {
    std::move(self()).throw_into(E(std::forward<Args>(args)...));
  }

 private:
  using promise_type = promise<expect::expected<T>, Tag>;
  promise_type& self() noexcept { return static_cast<promise_type&>(*this); }
  promise_type const& self() const noexcept {
    return static_cast<promise_type const&>(*this);
  }
};

/**
 * Create a new pair of init_future and promise with value type `T`.
 * @tparam T value type
 * @return pair of init_future and promise.
 * @throws std::bad_alloc
 */
template <typename T, typename Tag = default_tag>
auto make_promise() -> std::pair<future<T, Tag>, promise<T, Tag>> {
  auto start = new detail::continuation_start<T>();
  return std::make_pair(future<T, Tag>{start}, promise<T, Tag>{start});
}

template <typename>
struct is_tuple : std::false_type {};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T>
inline constexpr auto is_tuple_v = is_tuple<T>::value;

template <typename F, typename T>
struct apply_result;
template <typename F, typename... Ts>
struct apply_result<F, std::tuple<Ts...>> : std::invoke_result<F, Ts...> {};
template <typename F, typename T>
using apply_result_t = typename apply_result<F, T>::type;

template <template <typename...> typename T, typename, typename>
struct is_applicable;
template <template <typename...> typename T, typename F, typename... Ts>
struct is_applicable<T, F, std::tuple<Ts...>> : T<F, Ts...> {};
template <template <typename...> typename T, typename F, typename Tup>
inline constexpr auto is_applicable_v = is_applicable<T, F, Tup>::value;

template <typename... Ts, template <typename> typename Fut, typename Tag>
struct future_type_based_extensions<std::tuple<Ts...>, Fut, Tag>
    : detail::future_base_base<Tag, std::tuple<Ts...>, Fut> {
  using tuple_type = std::tuple<Ts...>;

  /**
   * Applies `std::get<Idx> to the result. All other values
   * are discarded.
   * @tparam Idx
   * @return
   */
  template <std::size_t Idx>
  auto get() {
    return std::move(self()).and_then(
        [](tuple_type&& t) noexcept -> std::tuple_element_t<Idx, tuple_type> {
          return std::move(std::get<Idx>(t));
        });
  }

  /**
   * Transposes a init_future and a tuple. Returns a tuple of mellon awaiting the
   * individual members.
   * @return tuple of mellon
   */
  auto transpose() -> std::tuple<Fut<Ts>...> {
    return transpose(std::index_sequence_for<Ts...>{});
  }

  template <typename F, std::enable_if_t<std::is_nothrow_invocable_v<F, Ts&&...>, int> = 0,
      typename R = std::invoke_result_t<F, Ts&&...>>
  auto and_then_apply(F&& f) -> Fut<R> {
    return std::move(self()).and_then(
        [f = std::forward<F>(f)](std::tuple<Ts...>&& tup) mutable noexcept {
          return std::apply(f, std::move(tup));
        });
  }

  template <typename F, std::enable_if_t<std::is_nothrow_invocable_r_v<void, F, Ts&&...>, int> = 0,
      typename R = std::invoke_result_t<F, Ts&&...>>
  void finally_apply(F&& f) {
    std::move(self()).finally(
        [f = std::forward<F>(f)](std::tuple<Ts...>&& tup) mutable noexcept {
          std::apply(f, std::move(tup));
        });
  }

  /**
   * Like `transpose` but restricts the output to the given indices. Other
   * elements are discarded.
   * @tparam Is indexes to select
   * @return tuple of mellon
   */
  template <std::size_t... Is>
  auto transpose_some() -> std::tuple<Fut<std::tuple_element<Is, tuple_type>>...> {
    return transpose(std::index_sequence<Is...>{});
  }

 private:
  template <std::size_t... Is>
  auto transpose(std::index_sequence<Is...>) {
    std::tuple<std::pair<future<Ts, Tag>, promise<Ts, Tag>>...> pairs(
        std::invoke([] { return make_promise<Ts>(); })...);

    std::move(self()).finally(
        [ps = std::make_tuple(std::move(std::get<Is>(pairs).second)...)](auto&& t) mutable noexcept {
          (std::invoke([&] {
             std::move(std::get<Is>(ps)).fulfill(std::move(std::get<Is>(t)));
           }),
           ...);
        });

    return std::make_tuple(std::move(std::get<Is>(pairs).first)...);
  }

  using future_type = Fut<std::tuple<Ts...>>;

  future_type& self() noexcept { return static_cast<future_type&>(*this); }
};

}  // namespace mellon

#endif  // FUTURES_FUTURES_H
