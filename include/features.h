#pragma once
//
// features.h — compile-time feature flags.
//
// Each flag is an `inline constexpr bool`. Code paths gated on these flags
// via `if constexpr (features::FOO)` are eliminated at compile time when
// the flag is false — no branch, no instructions, no register pressure.
//
// Combined with the matching runtime flag in PPOConfig (e.g.
// `BetHistoryConfig::enabled`), this gives two-tier control:
//
//   compile-time (here)         — include or exclude the feature entirely
//   runtime (PPOConfig)         — toggle within compiled-in features
//
// In particular, when a flag here is `false`, the runtime flag becomes a
// no-op: the corresponding work simply isn't in the binary. Useful for
// (a) micro-benchmarking the "no plumbing" baseline, (b) shipping a
// stripped binary that physically can't run a feature, (c) sanity-
// checking that a feature is purely additive (flipping the constexpr
// off should never break a no-feature run).
//
// Defaults below match the current main.cpp configuration so a normal
// build sees no behaviour change. Flip a flag to `false` to strip.
//

namespace poker_ppo::features {

// Per-round normalised hand_indexer bucket features (4 floats appended
// to the static obs block). When OFF, the obs is 4 floats smaller, the
// per-step write loop is gone, and the indexer-norm cache lazy-init is
// dead-code-eliminated. There is no runtime counterpart — the feature
// is purely compile-time.
inline constexpr bool HAND_STRENGTH = true;

// Bet-history attention encoder: when ON, an attention block over the
// per-action token sequence is registered in the network and called on
// every forward pass; the env writes a [T mask | T*F tokens] tail block
// in every observation. When OFF, ALL of that vanishes from the binary
// — module registrations, encode_history call site, env-side write,
// and the corresponding obs tail. The runtime BetHistoryConfig::enabled
// flag still controls toggling within compile-time-enabled builds.
inline constexpr bool ATTENTION_ENCODER = true;

// Hand-engineered per-round summary block (4 rounds × 4 floats appended
// to the obs between the static features and the attention tail).
// Symmetric gating to ATTENTION_ENCODER: env-side write and net-side
// parse strip together when this is OFF. Runtime
// RoundSummaryConfig::enabled controls toggling within enabled builds.
inline constexpr bool ROUND_SUMMARY = true;

// Pool-conditioning value head: a placeholder for a future feature
// (the on-tree implementation has been reverted). Reserved here so the
// next reintroduction slots into the same pattern without churn.
inline constexpr bool POOL_CONDITIONING = false;

}  // namespace poker_ppo::features
