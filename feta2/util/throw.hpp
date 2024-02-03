#pragma once

#include <sstream>

/** @brief Macro used to throw exceptions. For host code only. */
#define FETA2_HTHROW(...)                                                      \
    ::feta2::err::detail::throw_(__FILE__, __LINE__, __func__, ##__VA_ARGS__)

#define FETA2_HASSERT(condition, ...)                                          \
    if (!(condition))                                                          \
    ::feta2::err::detail::throw_(__FILE__, __LINE__, __func__,                 \
        "assertion failed: (", #condition, ")\n\t ", ##__VA_ARGS__)

namespace feta2 {
namespace err {
namespace detail {

// Concatenate variable number of arguments into a string
template<typename... Ts>
std::string cat(Ts&&... args)
{
    // Uses C++17 fold expression
    std::ostringstream oss;
    (oss << ... << std::forward<Ts>(args));
    return oss.str();
}

template<typename... Ts>
void throw_(
    const std::string file, int line, const std::string func, Ts... params)
{
    throw std::runtime_error(
        cat("[", file, ":", line, ":", func, "] FETA error: ", params...));
}
} // namespace detail
} // namespace err
} // namespace feta2
