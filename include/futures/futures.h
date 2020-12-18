#ifndef FUTURES_FUTURES_H
#define FUTURES_FUTURES_H
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <iostream>

namespace futures {

namespace detail {

template <typename T>
struct no_deleter {
  void operator()(T*) const noexcept {}
};

template <typename T>
using fixed_ptr = std::unique_ptr<T, no_deleter<T>>;

struct invalid_pointer_type_inline_value {};
struct invalid_pointer_type_future_abandoned {};
struct invalid_pointer_type_promise_fulfilled {};

extern invalid_pointer_type_inline_value invalid_pointer_inline_value;
extern invalid_pointer_type_future_abandoned invalid_pointer_future_abandoned;
extern invalid_pointer_type_promise_fulfilled invalid_pointer_promise_fulfilled;

#define FUTURES_INVALID_POINTER_INLINE_VALUE(T) \
  reinterpret_cast<::futures::detail::continuation_base<T>*>(&detail::invalid_pointer_inline_value)
#define FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T) \
  reinterpret_cast<::futures::detail::continuation<T>*>(&detail::invalid_pointer_future_abandoned)
#define FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T) \
  reinterpret_cast<::futures::detail::continuation<T>*>(&detail::invalid_pointer_promise_fulfilled)

template <typename T>
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

  template<typename U = T, std::enable_if_t<std::is_move_constructible_v<U>, int> = 0>
  T&& cast_move() { return std::move(ref()); }
  template<typename U = T, std::enable_if_t<!std::is_move_constructible_v<U>, int> = 0>
  T& cast_move() { return ref(); }

 private:
  alignas(T) std::array<std::byte, sizeof(T)> value;
};

template <>
struct box<void> {
  box() noexcept = default;
  explicit box(std::in_place_t) noexcept {}
  void emplace() noexcept {}
};

template <typename T, typename = void>
struct small_box : box<void> {
  static constexpr bool stores_value = false;
};
template <typename T>
struct small_box<T, std::enable_if_t<sizeof(T) <= 32>> : box<T> {
  using box<T>::box;
  static constexpr bool stores_value = true;
};

}  // namespace detail

template <typename T, typename = void>
struct future;
template <typename T>
struct promise;


namespace handlers {

template<typename T>
struct allocation_failure {
  void operator()(future<T>) noexcept { std::abort(); }
};

}

namespace detail {

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

template <typename T>
void abandon_continuation(continuation_base<T>* base) noexcept {
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T),
                                           std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    if (expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
      base->destroy();
      delete base;
    }
  }
}

template <typename T, typename... Args>
auto allocate_frame_noexcept(Args&&... args) noexcept -> T* {
  auto frame = new (std::nothrow) T(std::forward<Args>(args)...);
  if (frame == nullptr) {
    std::abort();
  }
  return frame;
}

inline void hard_assert(bool x) {
  if (!x) {
    std::abort();
  }
}

template <typename T, typename F, typename G>
void insert_continuation_final(continuation_base<T>* base, G&& f) noexcept {
  static_assert(std::is_nothrow_invocable_r_v<void, F, T&&>);
  if (base->_next.load(std::memory_order_acquire) ==
      FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
    // short path
    std::invoke(std::forward<G>(f), base->cast_move());
    base->destroy();
    delete base;
    return;
  }

  auto step =
      detail::allocate_frame_noexcept<continuation_final<T, F>>(std::in_place,
                                                                std::forward<G>(f));
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, step, std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    hard_assert(expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T));
    std::invoke(step->function_self(), base->cast_move());
    base->destroy();
    delete base;
    delete step;
  }
}

template <typename T, typename F, typename R, typename G>
auto insert_continuation_step(continuation_base<T>* base, G&& f) -> future<R> {
  static_assert(std::is_nothrow_invocable_r_v<R, F, T&&>);
  if (base->_next.load(std::memory_order_acquire) ==
      FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T)) {
    // short path
    auto fut =
        future<R>{std::in_place, std::invoke(std::forward<G>(f), base->cast_move())};
    base->destroy();
    delete base;
    return std::move(fut);
  }

  auto step =
      detail::allocate_frame_noexcept<continuation_step<T, F, R>>(std::in_place,
                                                                  std::forward<G>(f));
  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, step, std::memory_order_release,
                                           std::memory_order_acquire)) {  // ask mpoeter
    hard_assert(expected == FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T));
    if constexpr (future<R>::is_value_inlined) {
      auto fut = future<R>{std::in_place,
                           std::invoke(step->function_self(), base->cast_move())};
      base->destroy();
      delete base;
      delete step;
      return std::move(fut);
    } else {
      step->emplace(std::invoke(step->function_self(), base->cast_move()));
      base->destroy();
      delete base;
    }
  }

  return future<R>{step};
}

template <typename T, typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
void fulfill_continuation(continuation_base<T>* base,
                          Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args...>) {
  base->emplace(std::forward<Args>(args)...);

  continuation<T>* expected = nullptr;
  if (!base->_next.compare_exchange_strong(expected, FUTURES_INVALID_POINTER_PROMISE_FULFILLED(T),
                                           std::memory_order_release,
                                           std::memory_order_acquire)) {
    if (expected != FUTURES_INVALID_POINTER_FUTURE_ABANDONED(T)) {
      std::invoke(*expected, base->cast_move());
    }
    base->destroy();
    delete base;
  }
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
  explicit function_store(std::in_place_t, G&& f) : Func(std::forward<G>(f)) {}

  [[nodiscard]] Func& function_self() { return *this; }
  [[nodiscard]] Func const& function_self() const { return *this; }
};

template <typename T, typename F, typename R>
struct continuation_step final : continuation_base<R>, function_store<F>, continuation<T> {
  static_assert(std::is_nothrow_invocable_r_v<R, F, T&&>);

  template <typename G = F>
  explicit continuation_step(std::in_place_t, G&& f)
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
  explicit continuation_final(std::in_place_t, G&& f)
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

}  // namespace detail

template <typename T>
struct promise {
  ~promise() { detail::hard_assert(_base == nullptr); }

  promise(promise const&) = delete;
  promise& operator=(promise const&) = delete;
  promise(promise&& o) noexcept = default;
  promise& operator=(promise&& o) noexcept = default;

  explicit promise(detail::continuation_start<T>* base) : _base(base) {}

  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  void fulfill(Args&&... args) && noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    detail::fulfill_continuation(_base.get(), std::forward<Args>(args)...);
    _base.reset();  // we can not use _base.release() because the constructor of T could
    // throw an exception. In that case the promise has to stay active.
  }

 private:
  detail::fixed_ptr<detail::continuation_start<T>> _base = nullptr;
};

template <typename T, typename F, typename R>
struct future_temporary : detail::function_store<F>, detail::small_box<T> {
  static constexpr auto is_value_inlined = detail::small_box<T>::stores_value;

  future_temporary(future_temporary const&) = delete;
  future_temporary& operator=(future_temporary const&) = delete;
  future_temporary(future_temporary&& o) noexcept : _base(nullptr) {
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::box<T>::emplace(o.cast_move());
        o.destroy();
      }
    }
    std::swap(_base, o._base);
  }
  future_temporary& operator=(future_temporary&& o) noexcept {
    detail::hard_assert(_base == nullptr);
    if (o.holds_inline_value()) {
      detail::box<T>::emplace(o.cast_move());
      o.destroy();
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
        return future_temporary<T, decltype(composition), S>(std::in_place,
                                                             std::move(composition),
                                                             detail::box<T>::cast_move());
      }
    }

    return future_temporary<T, decltype(composition), S>(std::move(composition),
                                                         std::move(_base));
  }

  template <typename G, std::enable_if_t<std::is_nothrow_invocable_r_v<void, G, R&&>, int> = 0>
  void finally(G&& f) {
    auto composition =
        detail::compose(std::forward<G>(f),
                        std::move(detail::function_store<F>::function_self()));

    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        std::invoke(composition, detail::box<T>::cast_move());
        detail::box<T>::destroy();
        _base.reset();
        return;
      }
    }

    return detail::insert_continuation_final<T, F>(_base.release(), std::move(composition));
  }

  void abandon() && { std::move(*this).finalize().abandon(); }

  operator future<R>() && { return std::move(*this).finalize(); }

  auto finalize() && -> future<R> {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        auto f = future<R>(std::in_place,
                           std::invoke(detail::function_store<F>::function_self(),
                                       detail::box<T>::cast_move()));
        detail::box<T>::destroy();
        return f;
      }
    }

    return detail::insert_continuation_step<T, F, R>(
        _base.release(), std::move(detail::function_store<F>::function_self()));
  }

  [[nodiscard]] bool holds_inline_value() const {
    return is_value_inlined && _base.get() == FUTURES_INVALID_POINTER_INLINE_VALUE(T);
  }

 private:
  template <typename G = F>
  future_temporary(G&& f, detail::fixed_ptr<detail::continuation_base<T>> base)
      : detail::function_store<F>(std::in_place, std::forward<G>(f)),
        _base(std::move(base)) {}
  template <typename G = F, typename S = T>
  future_temporary(std::in_place_t, G&& f, S&& s)
      : detail::function_store<F>(std::in_place, std::forward<G>(f)),
        detail::small_box<T>(std::in_place, std::forward<S>(s)),
        _base(FUTURES_INVALID_POINTER_INLINE_VALUE(T)) {}

  template<typename, typename>
  friend class future;
  template<typename, typename, typename>
  friend class future_temporary;

  detail::fixed_ptr<detail::continuation_base<T>> _base;
};


template <typename T, typename>
struct future : private detail::small_box<T> {
  static_assert(!std::is_void_v<T>, "void is not supported, use std::monostate instead");
  static constexpr auto is_value_inlined = detail::small_box<T>::stores_value;

  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_move_assignable_v<T>);

  future(future const&) = delete;
  future& operator=(future const&) = delete;
  future(future&& o) noexcept : _base(nullptr) {
    if constexpr (is_value_inlined) {
      if (o.holds_inline_value()) {
        detail::box<T>::emplace(o.cast_move());
        o.destroy();
      }
    }
    std::swap(_base, o._base);
  }
  future& operator=(future&& o) noexcept {
    detail::hard_assert(_base == nullptr);
    if (o.holds_inline_value()) {
      detail::box<T>::emplace(o.cast_move());
      o.destroy();
    }
    std::swap(_base, o._base);
  }

  ~future() { detail::hard_assert(_base == nullptr); }

  explicit future(detail::continuation_base<T>* ptr) noexcept : _base(ptr) {}

  template <typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
  explicit future(std::in_place_t, Args&&... args) noexcept(
      std::conjunction_v<std::is_nothrow_constructible<T, Args...>, std::bool_constant<is_value_inlined>>) {
    if constexpr (is_value_inlined) {
      detail::box<T>::emplace(std::forward<Args>(args)...);
      _base.reset(FUTURES_INVALID_POINTER_INLINE_VALUE(T));
    } else {
      _base.reset(new detail::continuation_base<T>(std::in_place,
                                                   std::forward<Args>(args)...));
    }
  }

  template <typename F, std::enable_if_t<std::is_nothrow_invocable_v<F, T&&>, int> = 0,
            typename R = std::invoke_result_t<F, T&&>>
  auto and_then_temp(F&& f) && noexcept {
    return future_temporary<T, F, R>(std::forward<F>(f), std::move(_base));
  }

  template <typename F, std::enable_if_t<std::is_nothrow_invocable_v<F, T&&>, int> = 0,
            typename R = std::invoke_result_t<F, T&&>>
  auto and_then(F&& f) && noexcept -> future<R> {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        auto fut = future<R>{std::in_place,
                             std::invoke(std::forward<F>(f), this->cast_move())};
        detail::box<T>::destroy();
        _base.reset();
        return std::move(fut);
      }
    }
    return detail::insert_continuation_step<T, F, R>(_base.release(), std::forward<F>(f));
  }

  template <typename F, std::enable_if_t<std::is_nothrow_invocable_r_v<void, F, T&&>, int> = 0>
  void finally(F&& f) {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        std::invoke(std::forward<F>(f), detail::box<T>::cast_move());
        detail::box<T>::destroy();
        _base.reset();
        return;
      }
    }

    return detail::insert_continuation_final<T, F>(_base.release(), std::forward<F>(f));
  }

  [[nodiscard]] bool holds_inline_value() const {
    return is_value_inlined && _base.get() == FUTURES_INVALID_POINTER_INLINE_VALUE(T);
  }

  void abandon() && {
    if constexpr (is_value_inlined) {
      if (holds_inline_value()) {
        detail::box<T>::destroy();
      }
    } else {
      detail::abandon_continuation(_base.release());
    }
    _base.reset();
  }

 private:
  detail::fixed_ptr<detail::continuation_base<T>> _base;
};


template<typename T>
struct future<T, std::enable_if_t<std::is_void_v<T>>> {
  template<typename S>
  struct always_false : std::false_type {};
  static_assert(always_false<T>::value, "void is not supported, use std::monostate instead");
};


template <typename T>
auto make_promise() -> std::pair<future<T>, promise<T>> {
  auto start = new detail::continuation_start<T>();
  return std::make_pair(future<T>{start}, promise<T>{start});
}

}  // namespace futures

#endif  // FUTURES_FUTURES_H
