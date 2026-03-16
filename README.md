# Multi-Agent Table Tennis (DDPG)

Two DDPG (Deep Deterministic Policy Gradient) agents trained to play table tennis in a Unity ML-Agents environment. Each agent learns continuous paddle control through reinforcement learning with shared experience and centralized training.

## Setup

1. **Clone the repository** (or download and extract).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Unity Tennis environment**  
   The training notebook expects the Tennis environment at one of:
   - `Tennis_Windows_x86_64/Tennis_Windows_x86_64/Tennis.exe`
   - `Tennis_Windows_x86_64/Tennis.exe`  

   If you have the environment as a zip (e.g. from Udacity DRLND or a similar course), unzip it into the project root so that `Tennis.exe` is at one of the paths above. The executable must match the `unityagents` Python package version (legacy 0.4.x) to avoid connection issues.

## Running

- **Train / experiment**: Open `Tennis (2) (1).ipynb` in Jupyter and run the cells. The notebook loads the environment, creates a single shared `DDPGAgent`, and runs the multi-agent training loop.
- **Code layout**: The `deep_rl` package contains the DDPG agent (`deep_rl.agent.DDPG_agent`) and actor-critic networks (`deep_rl.network.network`). The notebook imports `DDPGAgent` from `deep_rl.agent.DDPG_agent`.

## Project structure

- `deep_rl/` – DDPG agent, Actor/Critic networks, replay buffer, OU noise
- `Tennis (2) (1).ipynb` – Training and evaluation notebook
- `requirements.txt` – Python dependencies
- `ML_Project_Report_MultiAgentTableTennis-1 (1).pdf` – Project report (if included)

Checkpoints (`*.pth`) and the Unity environment zip are listed in `.gitignore` to keep the repo small; add them locally or host them elsewhere if needed.
