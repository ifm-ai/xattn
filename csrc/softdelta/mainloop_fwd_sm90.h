#pragma once

namespace xattn {
namespace ops {

template <typename Mainloop>
struct FlashSoftDeltaMainloopFwdSm90 : Mainloop {};

template <typename Mainloop>
struct FlashSoftDeltaMainloopAdapterFwdSm90 {
  using Type = FlashSoftDeltaMainloopFwdSm90<Mainloop>;
};

}  // namespace ops
}  // namespace xattn
