# TD-MPC2-LQ structural design

Status: proposed research variant, 18 November 2026. This specifies a new linear/quadratic adaptation; it is not a claim of equivalence to standard TD-MPC2. No comparative experiment or runtime result is presented here.

## 1. Purpose and naming

Keep the five-method project: DDPG, TD3, PPO, SAC and TD-MPC2. For the scalar LQR stage, introduce **TD-MPC2-LQ**, a structured model-learning and planning variant inspired by the pinned official TD-MPC2 implementation. Compare its learning behavior with the existing structured PPO. Report it under the modified name, with standard neural TD-MPC2 identified separately wherever its earlier results are shown.

The intended savings come from replacing neural models with a handful of coefficients and vectorizing arithmetic across seeds. Do not replace the learner with a Riccati controller, solve the true Bellman equation analytically during training, or use the exact optimum to choose actions. Those would answer a different research question.

Compute becoming negligible is a design objective, not an established result. A measured throughput gate must precede a 40-seed million-step allocation.

## 2. Match the physical problem

For the first experiment use x_next = x + u, x0 = 1, observed cost c = x² + u² and gamma = 0.99. The learner is told the linear/quadratic model families, but not the true coefficients or the optimal gain. The environment supplies state transitions and scalar rewards. Convert rewards to costs internally using c = -reward. This guide uses cost minimization throughout.

The evaluator alone computes the discounted Riccati optimum and g = 1 - J*/J. Do not use the gap, exact gain or model parameters in training, initialization, early stopping or actor fitting.

The structured observation is z = x: no learned encoder, hidden velocity or extra state is provided. For a later LQI experiment, use a separately specified augmented state and matching objective. This document does not authorize transferring the scalar design's guarantees to nonlinear tracking tasks.

## 3. Small learned components

Define psi(x,u) = [x², 2xu, u², 1]. Use raw physical coordinates and float64 arithmetic.

| Component | Parameterization | Scalar parameter count |
|---|---|---:|
| Encoder | z = x, fixed identity | 0 |
| Transition model | f_hat(x,u) = a_hat x + b_hat u | 2 |
| Stage-cost model | c_hat(x,u) = w_c · psi(x,u) | 4 |
| Two cost critics | C_j(x,u) = w_j · psi(x,u) | 8 |
| Policy prior | u distributed as Normal(-k x, sigma²) | 2 |
| Total online learned coefficients | Excludes optimizer and target copies | 16 |

The constants allow predictions of continuing cost under stochastic actions; a critic restricted to a homogeneous quadratic would omit this contribution. The dynamics model assumes a zero equilibrium and no affine disturbance, which is privileged structural knowledge and must be disclosed. Do not initialize a_hat and b_hat to the true plant coefficients. Initialize them to zero; initialize cost and critic coefficients to small independent random values (standard deviation 0.01), k = 0 and sigma = 0.5. These are proposed frozen settings, not validated defaults.

Do not project learned coefficients onto the Riccati solution. A quadratic critic is linear in its coefficients, but is not necessarily a convex function of u. The sampling planner below does not require positive curvature. If nonfinite predictions arise, mark and account for the run rather than silently repairing its controller using the true plant.

## 4. Preserve joint multi-step learning

Replay stores real contiguous trajectories, including the final observation at time-limit truncation. Sample length-H transition sequences that never cross resets. Training episodes reset after 500 interactions, but these resets are not true terminals for the continuing discounted objective.

For each sampled sequence, start x_hat_0 = x_0 and recursively predict x_hat_(h+1) = f_hat(x_hat_h,u_h), using the stored real actions. Use the same multi-step weighting rho^h with rho = 0.5.

Define these losses, averaged over the batch and relevant time/critic axes:

    L_model = average_h rho^h (x_hat_(h+1) - x_(h+1))²
    L_cost  = average_h rho^h (c_hat(x_hat_h,u_h) - c_h)²
    y_h     = c_h + gamma max_j C_target,j(x_(h+1),u'_h)
    L_value = average_(h,j) rho^h (C_j(x_hat_h,u_h) - stop_gradient(y_h))²

Here u'_h is sampled from the current policy prior at the real next state, with the action convention in Section 7. Detach target calculations. Taking the maximum of cost critics is the sign-consistent counterpart of the official minimum of reward Q critics. A true terminal masks the continuation; a time-limit truncation does not.

Optimize L = 20 L_model + 0.1 L_cost + 0.1 L_value jointly. In particular, differentiate through the predicted-state recursion for the cost and critic losses, as well as for model consistency. Independently fitting a linear model by least squares and attaching an analytic LQR controller is not this algorithm.

Use explicit analytic derivatives or a small automatic-differentiation reference implementation, with the NumPy derivative path checked against it or finite differences. Freeze targets during each update. After updating model/critics, update the actor, then apply Polyak target updates with tau = 0.01. Use Adam with learning rate 0.0003 and gradient-norm clipping at 20. Document the exact averaging conventions because they affect effective learning rates.

Squared scalar regression replaces TD-MPC2's distributional two-hot prediction losses and symlog representation. That is a substantial algorithmic change, not just a smaller network. Its effects cannot be assumed harmless. Preserve the physical training rewards without clipping; any later reward normalization is a separately named ablation.

## 5. Retain a learned policy prior

For predicted states detached from the model update, minimize

    L_actor = average_h rho^h [mean_j C_j(x_hat_h,u_theta) / S - alpha H(pi_theta)]

with alpha = 0.0001 and u_theta = -k x_hat_h + sigma epsilon. S is a detached positive scale: retain the official moving 5th-to-95th-percentile spread rule, with a lower bound of 1 and moving-average rate 0.01. Apply the appropriate chain rule if the bounded action map is selected.

This actor is a proposal generator for planning and a source of bootstrap actions, not the deployed controller by itself. Setting it equal to the exact LQR gain, using an analytic critic minimizer instead of learning it, or scoring only its mean would change the experiment.

This prior update is not PPO: it uses off-policy replay and critic-guided differentiation rather than an on-policy clipped probability-ratio objective.

## 6. Retain short-horizon sampling control

At every real action, optimize a short action sequence using the learned model, learned stage costs and a terminal critic. For a candidate sequence, predict states recursively and score

    score = sum_(h=0 to H-1) gamma^h c_hat(x_hat_h,u_h)
            + gamma^H mean_j C_j(x_hat_H,u_prior).

Sample u_prior from the actor at the predicted terminal state. Use the mean of both critics for planning; the maximum is used for bootstrap targets. These roles match the reward-sign interpretation of the official implementation.

Retain the official planning pattern: mix actor-generated trajectories with random candidates, select elites, weight lower-cost elites more heavily, update the sequence mean and spread, and repeat. Execute only the first action, observe the real next state, shift the planned sequence for a warm start and replan. Do not roll out the true dynamics inside the learner's search.

For costs s_i among elites, use normalized weights exp[-T(s_i - min s)], T = 0.5. The deployed planner samples an elite sequence according to these weights. Training adds exploration noise to its first action; evaluation removes that extra noise but retains candidate, prior and elite-selection randomness. Reset the warm start at each evaluation start.

The primary adaptation retains sampling rather than analytically solving a finite-horizon QP. With the critic maximum in targets, random prior and unconstrained fitted quadratics, it is misleading to describe the whole learning problem as a single convex program. An analytic MPC alternative can be a future ablation, but it should not silently replace this planner.

## 7. Resolve action constraints before the comparison

The existing PPO mean controller is unbounded, whereas the official TD-MPC2 planner uses bounded actions. Two coherent protocols are available; neither should be mixed silently with the other.

**Primary structured-LQR proposal:** use unbounded physical actions for both learners. The linear-Gaussian prior, finite sampled action sequences and learned quadratic functions are mutually compatible. Preserve finite lower/upper proposal standard deviations (0.05 and 2) for numerical conditioning, but do not clip physical states or inputs. This removes the official action-squashing/clipping machinery and is an explicit modification. It also removes the earlier pilot's universal bounded-action evaluation-tail guarantee. Predicted cost may be misleading or unbounded below outside the visited region; the finite candidate planner does not remove that risk.

**Bounded sensitivity experiment:** retain [-2,2] actions, normalized internal planning coordinates and a correspondingly bounded actor. The actor then is a bounded linear-mean prior, not a globally linear control law. Use the same physical bounds for PPO and explicitly reevaluate its deployment map. A clipped linear PPO controller no longer has a globally exact linear Lyapunov cost formula. The earlier PPO report is historical context rather than a matched result under the changed protocol.

The first protocol is closer to the original PPO benchmark's action class. The second is closer to standard TD-MPC2 and permits bounded-action tail certification. The next experiment must explicitly select a protocol and rerun PPO consistently. The fact that the exact optimal trajectory fits within [-2,2] does not by itself make the two action classes identical during training or off that trajectory.

## 8. Two compute configurations and what each means

Separate representation reduction from reductions in search or update effort.

| Setting | Structure-only reference | Proposed fast primary |
|---|---:|---:|
| Prediction horizon H | 3 | 3 |
| Candidates per planner iteration | 512 | 64 |
| Planner iterations | 6 | 3 |
| Elites | 64 | 8 |
| Actor-prior trajectories | 24 | 8 |
| Replay batch size | 256 | 64 |
| Model/critic/actor update | Every interaction | Every 16 interactions |
| Target update | After each learner update | After each learner update |

Both configurations use the same 16-coefficient model family. The fast version forecasts 576 scalar transitions per real action instead of 9,216 neural latent transitions in the original planner. Batched NumPy operations across candidates and independent seeds avoid a Python loop per candidate and remove the need for a GPU. Updating every 16 interactions is a separate algorithmic change and may affect sample efficiency; the same numerical tau now represents a slower rate per environment interaction.

Use a fixed 2,500-interaction random-collection warm-up for both structured variants. At warm-up completion, perform 2,500 learner updates for the structure-only reference and ceil(2500/16) updates for the fast version, then follow their stated update schedules. Store at most one million transitions per seed with reproducible sequence sampling. Use separate initialization, collection, replay and evaluation RNG streams. A complete reset policy for dead/nonfinite runs must be declared before launch; never reset a failed seed and report only its replacement.

These are proposed engineering settings, not empirically optimized values. Predeclare them. Tune only in a separate development study if necessary, and do not label a tuned configuration as the original untuned pilot. Compare fast versus structure-only in a small ablation before interpreting any difference against PPO as purely a representation effect.

## 9. Runtime acceptance gate

Count environment interactions, planner predictions, optimizer updates and elapsed time separately. Time a short execution-only batch first without choosing settings based on return. Then time a representative 100,000-interaction run after warm-up to capture replay and memory behavior; this is a throughput benchmark, not the final multi-seed comparison.

For example, if 40 seeds are processed together at 100,000 interactions per seed in t seconds, estimate the training time for 40 x one million as approximately 10t, then add separately measured checkpoint evaluation and output time. Check memory and scaling rather than assuming a linear extrapolation.

Proposed usability target: the full training allocation should fit within roughly two hours on the available machine, with evaluation included in the measured budget or reported separately. This is a project target, not a prediction. If the implementation misses it, profile vectorization and storage first. Changes in model size, candidates, horizon or update ratio require a new named configuration; do not silently alter them after inspecting comparison outcomes.

Even the fast version involves billions of elementary operations across 40 million interactions. It should be far cheaper than the neural implementation, but “compute time is a non-issue” cannot be promised from parameter count alone. Do not reuse the earlier neural runtime estimate for this variant.

## 10. Evaluation that does not assume the answer

Primary performance is the discounted physical cost of the **deployed planner**, not its actor and not an effective linear-gain fit. Average a common, fixed planner evaluation suite within each trained controller; then summarize across training runs. Make clear whether inference is conditional on that suite or also models new planner randomness. Checkpoint scoring must restore training RNG and planner memory.

For bounded actions on this scalar plant, retain the existing analytical tail envelope. For unbounded planning, no such envelope follows merely from gamma^H being small. Either derive and verify a suitable policy-growth bound or report a matched finite-horizon experiment as the primary observable. In the latter case, calculate the exact finite-horizon LQR reference with the same terminal cost, score both methods over that horizon, and label the gap finite-horizon. Repeating rollouts for 2H is a sensitivity check, not an infinite-horizon certificate.

If infinite-horizon comparison with the old PPO report is essential, resolve the planner-tail question before running the final experiment. Do not report a truncated infinite-horizon gap as exact or substitute the fitted gain's Lyapunov cost. Keep old PPO numbers separate from fresh matched-protocol results.

The sampled planner may have memory and stochastic actions even when its components are linear or quadratic. Its closed loop is generally not the linear map x_next = (1-k)x. Therefore 0<k<2 applied to its actor or fitted gain is not a certificate for the planner. Report clearly defined rollout failures, tracking residuals and action growth separately from mathematical stability claims.

With only a few checkpoints, report a late-budget average as descriptive; do not call it a proven limiting floor. Sample extrema describe measured runs, not the support of all possible seed outcomes. Bound any score before applying concentration inequalities and identify numerical truncation error explicitly.

## 11. Validation before a scientific run

1. Check the real environment and Riccati evaluator against the scalar closed forms. Isolate them from learner access.
2. Check quadratic features, model recursion and every gradient against finite differences, including signs of cost critics and all multi-step model derivatives.
3. Check replay sequence alignment, no crossing of resets, true-terminal masking and time-limit bootstrapping.
4. Check that cost pessimism uses max, elite selection prefers smaller predicted costs, actor proposals affect candidates, terminal critic changes planning, and target updates follow the declared schedule.
5. Check seed independence from batching: the same seed should produce the same result alone and beside others. Check evaluation restores RNG and planner warm starts.
6. Use synthetic known quadratics in unit tests to validate planner behavior, without giving such oracle information to a trained run. Reject tests that simply confirm a function against itself.
7. Log nonfinite parameters, predicted and real costs, action magnitudes, clipping if applicable, model errors, effective gains and numerical failures. Preserve failed runs.
8. Validate cost evaluation under the selected bounded or unbounded protocol. Compare against an exact finite-horizon solution where that is the claimed reference.
9. Benchmark throughput and memory before approving a large allocation. Run learning smoke checks as validation, but record any resulting parameter changes as development rather than pretend the configuration was never examined.

## 12. What distinguishes this from the other four methods

| Method | Learned object driving the deployed action | Distinctive operational choice |
|---|---|---|
| Structured PPO | Direct linear-Gaussian policy | On-policy clipped-ratio learning |
| Structured DDPG | Direct deterministic linear actor | Replay-based critic-guided actor updates |
| Structured TD3 | Direct deterministic linear actor | Twin critics, delayed actor updates, target smoothing |
| Structured SAC | Direct stochastic policy | Entropy-regularized off-policy learning |
| TD-MPC2-LQ | Planner using a learned model and terminal critics | Multi-step predictive consistency, actor-prior proposals and receding-horizon search |

This table describes the intended comparison, not implementations already validated for all five. Each requires its own explicit adaptation specification. Similar small function classes do not make the methods identical; the data use, targets, exploration and deployed decision procedures remain different.

## 13. Allowed conclusions and claim limits

A future experiment can compare the learning efficiency and consistency of these named structured implementations on a matched linear-quadratic task. It cannot establish how standard neural TD-MPC2 ranks on general continuous control, prove this adaptation preserves TD-MPC2's empirical strengths, or make novelty claims without a literature review. The broader term “linear model-predictive control with learned terminal values” also applies; the modified name documents lineage rather than establishing a new algorithmic category.

Before the next comparison, implement this specification, select and freeze the action/evaluation protocol, pass the validation gate, and measure runtime. No 40-seed or million-step comparison is launched by this document.

## Source basis

The structural comparison uses the locally inspected official TD-MPC2 source at commit 332f19e4aba4bffe2dc736f45ea7f7bdc585903a, particularly `tdmpc2.py`, `common/world_model.py` and `config.yaml`, and the project PPO/LQR snapshot at a193ef9. Official source: https://github.com/nicklashansen/tdmpc2/tree/332f19e4aba4bffe2dc736f45ea7f7bdc585903a

All equations and choices for TD-MPC2-LQ above are a proposed adaptation. They have not been presented as results or guarantees of the original authors.
