#pragma once
//
// features.h — compile-time feature flags. Two-tier control with
// PPOConfig: the constexpr below decides what's in the binary; the
// runtime flag toggles within compiled-in features. Setting a flag here
// to false makes the runtime flag a no-op.
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

// VRPO-style privileged Q-critic (Liu et al. 2026, "GAE Falls Short").
// When ON: (1) the critic is an action-value head Q(s,a) instead of a
// scalar V, fed the OPPONENT's hole cards as a privileged obs tail block
// (centralised training, decentralised execution — the actor never sees
// it); (2) the advantage is the Expected-SARSA(λ) "Q-boosting" estimator
// Â = Q(s,a) − V̄(s) + Σ(λγ)^k δ⁺, δ⁺ = r + γV̄(s') − Q(s,a), with
// V̄(s) = Σ_a π(a|s)Q(s,a) — averaging over the action distribution
// removes the future-action sampling noise that inflates GAE's variance
// in imperfect-information self-play. When OFF the V-critic + GAE path is
// unchanged. The privileged tail and Q-head strip from the binary when OFF.
inline constexpr bool PRIVILEGED_Q_CRITIC = true;

}  // namespace poker_ppo::features
