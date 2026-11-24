# Bounded PPO versus TD-MPC2-LQ: locked protocol

Recorded locally on 23 November 2026, before the full comparison. This is a protocol record, not an externally registered preregistration.

The user selected **bounded actions for both methods with certified infinite-horizon rollout tails**. Run 40 training seeds, one million physical interactions per seed and method, and checkpoints at 0, 50,000, ..., 1,000,000. Draw the repository's evaluation seeds using `split_seeds(0, 40, 0).eval`. Training consumes no exact optimum, gap, or evaluation outcomes. No tuning seeds or outcome-dependent stopping are used. A preceding 100,000-step throughput run repeats the same frozen TD configuration and is excluded from the final sample; it is not additional independent evidence.

The plant is x_next=x+u, cost=x²+u², x0=1, gamma=.99. Clip applied actions to [-2,2]; never clip states or observed costs. The unconstrained Riccati optimum remains an exact constrained optimum **from this starting state**: its entire trajectory uses admissible actions. This is not a claim of a globally quadratic constrained value function.

## Controllers

PPO retains the repository defaults: eight environments, 250-step rollout/reset, ten epochs, four minibatches, lr=.0003, GAE lambda=.95, ratio clip=.2, value weight=.5, entropy weight=0, norm clip=.5, initial gain=0 and std=.5. Samples and likelihoods refer to the latent Gaussian; the actuator clips samples before computing physical transitions and rewards. Deployment is the clipped mean. The original quadratic value approximator is retained even though clipping removes global exact representability of the value function. The old unbounded PPO report is historical context, not a matched result.

TD-MPC2-LQ uses the 16-parameter structural design and its **fast** configuration: H=3, 64 candidates (eight actor trajectories replace eight random candidates), three iterations, eight elites, replay batch=64, one update every16 interactions after warmup. Initial dynamics coefficients=0; cost and critic coefficients iid Normal(0,.01); actor gain=0 and std=.5. Warmup=2,500 uniform random actions, followed by ceil(2500/16)=157 initial learner updates. Subsequently update at 2516,2532,... . Reset physical collection episodes every500 steps, without masking their bootstrap targets. Replay stores all collected transitions and samples uniformly among valid within-episode three-transition starts. Model/cost/critic weights=20/.1/.1, temporal weight=.5, Adam lr=.0003, gradient norm bound20, Polyak tau=.01. No analytic LQR/QP is used for action selection.

Use physical coordinates throughout, including planning. Actor proposals and their reparameterized gradients use actuator clipping. The entropy term is the latent Gaussian's entropy, not the differential entropy of the clipped action distribution; this is an explicit adaptation. Critic averaging includes both critics and H predicted stages; actor averaging includes H+1 detached predicted states. Keep per-seed percentile scales with EMA .01 and floor1. The stage-cost and critic losses propagate through the learned multi-step dynamics. Bootstrap cost targets use the maximum target critic; planning uses mean critics. Planner proposal std is clamped to [.05,2]; the learned actor's std is not projected. Evaluation retains sampling and warm starts within each rollout, but removes the additional training exploration perturbation.

The compiled planner receives random arrays from per-seed generators; it uses no plant coefficients. Numba changes execution speed, not the specified objective. Initialization, collection, replay and evaluation use separate streams. Within replay, sampling and target/actor draws share the replay generator. PPO uses its original generator layout. Training seed identifiers are shared across methods; their internal draws need not coincide.

These methods have equal environment-interaction allocations, not equal optimizer work or reset schedules. Differences against PPO cannot be attributed solely to model-based versus model-free learning, or solely to representation reduction. A separate small structure-only ablation uses the design's 512 candidates, 64 elites, 24 prior trajectories, six iterations, batch256 and update_every1; it is exploratory and not a 40-seed ranking.

## Evaluation and error control

At every checkpoint, TD evaluation uses planner streams 910000,910001,910002, initialized identically for every trained controller. Each starts at x0=1 and zero planner warm start. Average **costs** over the three streams, then compute g=1-J_star/mean_cost. PPO deployment is deterministic, so one rollout suffices. Report inference conditional on this fixed suite; three streams do not establish the expected performance over arbitrary deployment randomness. Preserve training RNGs and planner memory during evaluation.

After H simulated evaluation actions, |x_(H+j)| <= |x_H|+2j gives the pathwise omitted-cost envelope

    gamma^H [ (|x_H|²+4)/(1-gamma)
              +4|x_H|gamma/(1-gamma)²
              +4gamma(1+gamma)/(1-gamma)³ ].

Stop when this is below 1e-6 for every controller, with a 5,000-action safeguard. Use nonnegative stage costs to form [J_lower,J_upper] and the monotone interval for g. These are analytic truncation bounds evaluated in float64, not directed-rounding floating-point certificates. Since J>=J_star, the gap interval width is at most approximately 1e-6/J_star (a conservative allowance of 2e-6 is also valid). Do not fit a gain and substitute its Lyapunov cost for the planner's actual cost.

Training runs with nonfinite loss, model or target parameters are retained and penalized with gap1 from the failure checkpoint onward. Log first failure and preserve its parameters; subsequent numerical fallback states are only for batching. Their fallback rollout costs are not certified costs for the failed policy. Report failure counts separately. No replacement seeds or restarted learning. No early physical termination or alleged stability certificate from an effective fitted gain.

## Analysis

Primary endpoint: final checkpoint difference in mean normalized gap, TD minus PPO. Report seed-level means, sample standard errors, medians, ranges, and fraction with g<=.1. Across-checkpoint means and empirical ranges are descriptive learning curves. Prefix means illustrate averaging over training seeds; they do not prove asymptotic convergence. Effective gains and terminal states are diagnostics, not global stability results.

For a conservative final-only 95% interval on the population mean difference, use the without-replacement bounded-difference Hoeffding bound for scores in [-1,1], with half-width sqrt(2 log(40)/40), plus the truncation allowance; finite-population correction is negligible for 40 of 2^32 seeds. For individual-method means use sqrt(log(40)/(2*40)). Curves do not carry simultaneous confidence claims unless multiplicity across methods/checkpoints is included. A paired bootstrap interval may be shown only as an approximate descriptive inference with resampling explicitly documented, never as a distribution-free guarantee. Comparing sample maxima/minima is not a statement about all possible seeds.

## Validation and provenance

Require finite-difference checks of every multi-step model and actor derivative, known-quadratic planner checks, compiled-versus-Python arithmetic, replay reset alignment, seed batching invariance, evaluation state restoration, target updates, and agreement of certified rollouts with exact Riccati and zero-controller costs. The first validation suite passed 68 tests before the throughput allocation. Run manifests save configuration, seeds, runtime versions and SHA-256 hashes of learner/runner sources; save weights at every checkpoint, metrics, failure records and final RNG states.

Runtime acceptance target: approximately two hours including evaluation. Benchmark the representative 40-seed 100,000-step run before launching the full budget. Do not reduce search/update settings in response to measured returns. If runtime alone requires engineering changes, validate them and record them before restarting any final comparison.
