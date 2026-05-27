# Deconstructing Actor-Critic
Research code for the paper:

> **Deconstructing actor-critic: a large-scale empirical study of design components for practitioners**\
> Haseeb Shah, Lingwei Zhu, Adam White, Martha White.\
> *Under review at [PNAS](https://www.pnas.org).*

This study analyzes 33,000+ runs on a control task derived from a real drinking-water treatment plant, varying the lower-level design components of actor-critic algorithms such as the policy objective, policy parameterization, gradient estimator, and actor/critic update schedule to understand how each affects performance, run-to-run variability and hyperparameter sensitivity. The source data from the Drayton Valley water treatment plant in [`data/`](data/).

## Installation

Python 3.10+ is required (the codebase uses a few 3.10-only features).

```bash
pip install -r requirements.txt
```

## Database setup

Runs log to a MySQL server (local, cloud, or HPC). To configure credentials:

1. Copy [`configs/db/credentials.yaml`](configs/db/credentials.yaml) to `configs/db/credentials-local.yaml` and fill in your `username`, `password`, and `ip`. Files matching `credentials-*.yaml` are gitignored.
2. In [`configs/config.yaml`](configs/config.yaml) (or whichever config you launch), set `args.db_prefix` to your database-user prefix. On shared HPC systems such as the Digital Research Alliance of Canada (formerly Compute Canada), this usually has to match your account name; see their [database servers guide](https://docs.computecanada.ca/wiki/Database_servers). The database need not live on the same machine that runs the experiments: if it is remote, just ensure it accepts connections and provide its IP address in the config.

## Running a single experiment

```bash
python run_simulator.py +args.run=0
```

`args.run` is the primary key for the experiment in the database and must be supplied on the command line. In a multirun sweep, we assign each child run an id of `args.run + sweep_index`.

To override the agent or policy from the CLI:

```bash
python run_simulator.py +args.run=0 agent=ppo policy=beta
```

## Running a sweep

A sweep can be run using a config under `configs/0NN_config_*.yaml`. Each one defines a base agent, policy, and a sweeper-params grid. Together with an `args.seed` range, this defines the Cartesian product of runs:

```bash
python run_simulator.py --config-name=005_config_submitit +args.run=0
```

When running several sweeps in parallel against the same database, offset `args.run` by more than the size of any single sweep so the primary keys do not collide:

```bash
python run_simulator.py --config-name=005_config_submitit +args.run=10000 policy=gaussian
```

Alternatively, point each sweep at its own database by changing `args.db_name`.

The Hydra launcher is `joblib` by default (multi-process on one machine). For SLURM clusters, switch to the `submitit` launcher via the corresponding `005_config_submitit*.yaml`.

### Configuring the SLURM/submitit launcher

Before launching a sweep on your own cluster, edit the `hydra.launcher` section of the submitit config you intend to use (for example [`configs/005_config_submitit.yaml`](configs/005_config_submitit.yaml)):

- **`account`** is set to `???`. You must supply your SLURM allocation, either by editing the config or overriding it on the command line (`hydra.launcher.account=...`). On the Digital Research Alliance of Canada, this is something like `def-<pi>` or `rrg-<pi>`.
- **`setup`** is a list of shell commands run before each job. Replace `source /path/to/your/venv/bin/activate` with the path to your own virtual environment, and adjust the `module load` lines (e.g. `python/3.10`, `mariadb`) to match the modules available on your cluster.
- **Job resources** such as `timeout_min`, `mem_per_cpu`, `cpus_per_task`, `tasks_per_node`, `gpus_per_node`, and `array_parallelism` may be tuned to fit your allocation and the size of your sweep.


## Repository layout

```
├── run_simulator.py              # Entry point
├── agent_policy_factory.py       # Builds an (agent, actor, policy) triple from a Hydra config
├── agents/                       # Actor-critic agent implementations
│   ├── aclambda_agent.py         # Actor-Critic(λ)
│   ├── ddpg_agent.py             # DDPG
│   ├── greedyac_agent.py         # Greedy-AC
│   ├── mpo_adaptive_agent.py     # MPO
│   ├── ppo_agent.py              # PPO
│   ├── reinforce_agent.py        # REINFORCE
│   ├── sac_agent.py              # SAC
│   └── vmpo_agent.py             # V-MPO
├── models/
│   ├── actor_critic_nn.py        # Critic network
│   ├── tilecoding_mlp.py         # Tile-coded MLP
│   └── base_models/              # Policy parameterizations
│       ├── gaussian.py           # Gaussian
│       ├── squashed_gaussian.py  # tanh-squashed Gaussian
│       ├── student.py            # Student-t
│       ├── squashed_student.py   # tanh-squashed Student-t
│       ├── beta.py               # Beta
│       ├── softmax.py            # Discretized softmax
│       └── replay_buffer.py
├── simulators/
│   └── positional_pid_controller.py         # Backwashing-PID environment
├── data/                         # Source plant data the environment is derived from
├── experiment/                   # MySQL logging
├── kl_computation/               # KL utilities used in the mirror-descent variants
├── configs/                      # Hydra config tree 
│   ├── config.yaml               # Root default config
│   ├── 0NN_config_*.yaml         # Examples of configs for running different experiments
│   ├── agent/                    # Per-agent hyperparameters + sweeper grids
│   ├── policy/                   # Per-policy hyperparameters
│   ├── db/                       # MySQL credentials templates
│   └── schema/                   # Database logging schema
├── notebooks/                    # Example plotting notebook
└── requirements.txt
```