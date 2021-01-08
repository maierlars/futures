#ifndef FUTURES_ARANGODB_H
#define FUTURES_ARANGODB_H
#include <mellon/futures.h>

#define TRI_ASSERT(x)
#define THROW_ARANGO_EXCEPTION(x) throw (x);
#define TRI_ERROR_PROMISE_ABANDONED 1

namespace arangodb::futures {
struct arangodb_tag {};


template <typename T>
using Future = ::mellon::future<T, arangodb_tag>;
template <typename T>
using Promise = ::mellon::future<T, arangodb_tag>;

template<typename T>
using Try = ::expect::expected<T>;

template <typename T>
auto makePromise() {
  return ::mellon::make_promise<T, arangodb_tag>();
}

}  // namespace arangodb::mellon


template<>
struct mellon::tag_trait<arangodb::futures::arangodb_tag> {
  /* */
  struct assertion_handler {
    void operator()(bool condition) const noexcept {
      TRI_ASSERT(condition);
    }
  };


  template<typename T>
  struct abandoned_promise_handler {
    T operator()() const noexcept {
      // call arangodb crash handler for abandoned promise
      std::terminate();
    }
  };

  template<typename T>
  struct abandoned_promise_handler<::expect::expected<T>> {
    ::expect::expected<T> operator()() const noexcept {
      try {
        THROW_ARANGO_EXCEPTION(TRI_ERROR_PROMISE_ABANDONED);
      } catch (...) {
        return std::current_exception();
      }
    }
  };

  template<typename T>
  struct abandoned_future_handler {
    void operator()(T && t) const noexcept {
      TRI_ASSERT(false);
      if constexpr (::expect::is_expected_v<T>) {
        if (t.has_error()) {
          // LOG UNCAUGHT EXCEPTION AND CRASH
          std::terminate();
        }
      }
      // WARNING ??????
    }
  };

  static constexpr auto small_value_size = 1024;
  /* */
};


#endif  // FUTURES_ARANGODB_H
