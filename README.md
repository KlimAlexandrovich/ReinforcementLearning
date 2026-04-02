# Reinforcement Learning: Atari Breakout

This repository contains implementations and experiments with various Reinforcement Learning algorithms applied to the **Atari Breakout** environment (Gymnasium ALE).

## Project Overview

The goal of this project is to explore different RL architectures, understand their training dynamics, and implement a robust pipeline for experience collection, logging, and evaluation.

### Previous Experiments: DQN & DRQN

Initially, **Deep Q-Network (DQN)** and **Deep Recurrent Q-Network (DRQN)** were implemented and tested. Despite extensive hyperparameter tuning (adjusting learning rates, replay buffer sizes, epsilon decay, and network architectures like Dueling DQN), these models **did not yield satisfactory results** for the Breakout task.

The training results for these attempts are located at: `/Users/klimdajneko/Desktop/IT/Portfolio/ReinforcementLearning/breakout_logs`

**Why DQN/DRQN struggled with convergence:**
*   **Sample Efficiency vs. Stability:** DQN is off-policy and relies on a Replay Buffer. In a fast-paced environment like Breakout, the distribution of states changes rapidly, often leading to unstable Q-value estimations.
*   **Overestimation Bias:** Even with Double DQN, the model tended to overestimate the value of certain suboptimal actions, causing the agent to get "stuck" in local minima (e.g., staying in one corner).
*   **Sensitivity to Hyperparameters:** DQN is notoriously sensitive to the target network update frequency and the balance of exploration/exploitation.
*   **DRQN Complexity:** While DRQN adds a temporal dimension via LSTM/GRU, it significantly increases the training time and gradient instability when processing sequences of frames in a high-dimensional state space like Atari pixels.

---

## Current Approach: Proximal Policy Optimization (PPO)

Given the difficulties with value-based methods, the project transitioned to **PPO (Proximal Policy Optimization)**. PPO is an **on-policy** actor-critic algorithm that has become a gold standard in RL due to its balance of ease of implementation, sample efficiency, and robustness.

### The Mathematics Behind PPO

PPO's core innovation is the **clipped surrogate objective function**, which prevents the policy from changing too drastically in a single update step, ensuring stable training.

The objective function $L^{CLIP}$ is defined as:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right]$$

Where:
*   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policy.
*   $\hat{A}_t$ is the estimated advantage at time $t$.
*   $\epsilon$ is a hyperparameter (usually 0.1 or 0.2) that determines the clipping range.

This mechanism ensures that the policy update stays within a "trust region," making the learning process much more monotonic and reliable compared to vanilla Policy Gradient or DQN.

### Why stable-baselines3?

For the PPO implementation, the **stable-baselines3 (SB3)** library was chosen. The decision was motivated by:
1.  **Industry Standard:** SB3 is a widely used, well-tested, and documented library that implements RL algorithms with high reliability.
2.  **Learning Best Practices:** Integrating SB3 allowed for the exploration of professional RL workflows, including vectorized environments, custom callbacks, and modular logging.
3.  **Efficiency:** It provides highly optimized implementations of PPO, allowing us to focus on environment wrapping and hyperparameter strategy rather than low-level algorithmic debugging.

---

## Project Structure

*   `breakout_PPO.ipynb`: Main notebook for PPO training with detailed annotations.
*   `package/`: Custom modules for environment preprocessing, logging (`SmartLogger`), and SB3 utilities.
*   `ppo_train.py`: Script version of the PPO training pipeline.
*   `breakout_DQN.ipynb` / `breakout_DRQN.ipynb`: Historical records of value-based experiments.
*   `breakout_logs/`: Directory containing metrics, checkpoints, and recorded gameplay videos.

## Requirements

To install the necessary dependencies:
```bash
pip install -r requirements.txt
```