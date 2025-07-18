---
layout: post
title:  "MindGames (NeurIPS 2025 Competition)"
date:   2025-07-18
author: Leon Guertler & Bobby Cheng
---

**MindGames** is a a NeurIPS25 competition for text-based games that Bobby and I helped organize. Since we are obviously not allowed to participated, we wanted to write this short blog explaining how we would use [UnstableBaselines](https://github.com/LeonGuertler/UnstableBaselines), to train a model an submit it to the competition (optimally giving you a small edge over your competition).

MindGames has two main trackes focused on TheoryOfMindGames (one for 'Secretmafia-v0' and one for `Codenames-v0`, `ColonelBlotto-v0` and `ThreePlayerIPD-v0`). In our experience, the biggest challange is training small (i.e. <8B) models on games where the action output is part of the next players observation. In this blog we will focus on track two, since `ThreePlayerIPD-v0` is the only such environment in it.

Since we have also been training on these games for a couple of months, we created version of them where the observations are structured in a training-friendly manner (you can just use the suffic `-train`; i.e. `Codenames-v0` -> `Codenames-v0-train`). This will work both for offline training and your online submission.

First things first, we need to decide on a model and build the training script. In [Spiral](https://arxiv.org/pdf/2506.24119) we used the __Qwen3-4B-Base__ model so we can show the OOD math improvement of self-play. Since this worked very well, here we will use the slightly larger __Qwen3-8B-Base__.

As for the training script, since 8b is rather big, we will use UnstableBaselines, which uses LoRA and activation checkpointing, thus 48GB of vRam should be enough (for this blog we will use our 3x RTX6000 ada machine; if you only have access to 24gb of vRam you will likely have to go with the 4B version).

You can install UnstableBaselines via ```pip install unstable-rl``` (this will also install TextArena).

## Imports
Now we can set up the training script. I will first explain the components one by one, and then post the full script at the bottom.

First we need to import all relevant packages; specifically, **ray**, **unstable** (the package name for UnstableBaselines) and the reward transformations from UnstableBaselines.

```python

import ray, unstable, time 
import unstable.reward_transformations as retra
```

## Constants & Configs
Now we can build initialize constants and build the two necessary configuration configs (namely, the lora_config, specifying the lora size and which layers to apply it to, and the vllm_config, specifying out generation hyperparameters):
```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_GENERATION_LENGTH = 4096
MAX_TRAIN_SEQ_LEN = 3000 # if you are running out of vRam, you can decrease this.

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}
```

## Init Ray & Specify Environments
With everything imported and specified, we can now initialize ray and start building the relevant modules from unstable.
```python
ray.init(namespace="unstable") # the namespace is mostly important for the terminal_interface.py script (which loads the modules from the "unstable" namespace)

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            num_actors=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            num_actors=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            num_actors=3, 
            prompt_template="llama-instruct-zs"
        ),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            prompt_template="llama-instruct-zs"
        ),
])
```

The __TrainEnvSpec__ expects the **env_id** (same as in TextArena), the **num_players** of the environment, the **num_actors** we want to use to collect data (i.e. if num_players==num_actors we will use mirror self-play with no opponent sampling) and the **prompt_template** used to process the observations. The __EvalEnvSpec__ will use **fixed_opponent**=`google/gemini-2.0-flash-lite-001` as the default fixed opponent. Make sure to export your OpenRouter api key via `export OPENROUTER_API_KEY="YOUY_KEY"` before running the script.

## Tracker & Model Registry
Next up we will build the **Tracker** and **ModelRegistry**. The former is responsible for WandB logging as well as local path handling. The latter will keep track of all checkpoints and fixed opponents for model sampling.
```python
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))
```
We will add the base model (i.e. lora_path = None) to the model registry as well.

## ModelSampler & GameScheduler
Finally we can initialize our **ModelSampler** (since we are doing mirror self-play we will only use the base sampler here (i.e. where the __sample_opponent__ function is not implemented), using different opponent sampling methods to boost performance is a great starting off point) and **GameScheduler**. The **GameScheduler** is responsible for scheduling environments and models. In this example we will just randomly pick an environment for each game and don't need to sample opponents (since we are doing mirror self-player); but this is something worth playing with to boost performance.
```python
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(
    model_registry=model_registry
) 

game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
    model_sampler=model_sampler, 
    env_sampler=env_sampler, 
    logging_dir=ray.get(tracker.get_log_dir.remote())
)
```

## StepBuffer
In this code example we will use **REINFORCE** as the learning algorithm. Thus it is enough to keep track of individual __Steps__ (i.e. obs, action, reward triplets). However, if you want to use **A2C**, for example, you'll need to switch to the **TrajectoryBuffer**.
```python
step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
    max_buffer_size=768, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([
        retra.RoleAdvantageByEnvFormatter()
    ]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([
        retra.RewardForFormat(1.5), 
        retra.PenaltyForInvalidMove(1.0, -1.0)
    ]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([
        retra.NormalizeRewardsByEnv(True)
    ]),
)
```
The **max_buffer_size** specifies how many **Steps** at most will be held in the buffer. If we exceed this amount, the default sub-sampling strategy is to randomly delete old **Steps**. Additionally we are passing the **Tracker** object and specify the different reward transformations:
1. **FinalRewardTransforms** Will be applied first and is responsible for any transformation on the environment rewards. In this case we wil track the __ema__ role-based reward and subtract it from the final reward. This is very important for games that have strong role biases (i.e. TicTacToe).
2. **StepRewardTransforms** When the trajectory is split into individual steps, these transformations are applied. In this case we will reward the correct format (i.e. \boxed{}) and valid moves.
3. **SamplingRewardTransforms** Laslty, when pulling the next batch for training, the sampling rewards are applied. In this case just a standard normal transformation.

## Collector
Bringing it all together, we build the **Collector** which is responsible for running a fixed number of games in parallel, and pushing thinished episodes to the **StepBuffer**:
```python
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, 
    tracker=tracker, 
    buffer=step_buffer, 
    game_scheduler=game_scheduler
)
```

## Learner
As previously mentioned, in this example we will use the **REINFORCE** algorithm as our learner.
```python
learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH
))
```
To save vRam, we will use mini-batch-size 1. The other important vRam saving parameter is the activation_checkpointing. It will slow the code down by 20-30%, but also reduced vRAM requirements by almost 50%. If you are gpu-rich, you can disable it.

For the REINFORCELearner, we also need to set the **max_train_len** and **max_generation_length**. The latter is used for the Dr. GRPO trick. I.e.:
```python
seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
```
The **max_train_len** is used to truncate the number of tokens trained on from sequences that are too long without impacting the rewards. In practice the generation length of models usually spikes initally and then decreases. This parameter will allow you to train larger models w/o crashing the run and usually does not affect convergence.

## Putting it all together
The full script to run will be:
```python
import ray, unstable, time
import unstable.reward_transformations as retra

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_GENERATION_LENGTH = 4096
MAX_TRAIN_SEQ_LEN = 3000 # if you are running out of vRam, you can decrease this.

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}

ray.init(namespace="unstable") # the namespace is mostly important for the terminal_interface.py script (which loads the modules from the "unstable" namespace)

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            num_actors=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            num_actors=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            num_actors=3, 
            prompt_template="llama-instruct-zs"
        ),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            prompt_template="llama-instruct-zs"
        ),
])


tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

model_sampler = unstable.samplers.model_samplers.BaseModelSampler(
    model_registry=model_registry
) 

game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
    model_sampler=model_sampler, 
    env_sampler=env_sampler, 
    logging_dir=ray.get(tracker.get_log_dir.remote())
)

step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
    max_buffer_size=768, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([
        retra.RoleAdvantageByEnvFormatter()
    ]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([
        retra.RewardForFormat(1.5), 
        retra.PenaltyForInvalidMove(1.0, -1.0)
    ]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([
        retra.NormalizeRewardsByEnv(True)
    ]),
)

collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, 
    tracker=tracker, 
    buffer=step_buffer, 
    game_scheduler=game_scheduler
)

learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH
))

try:
    collector.collect.remote(
        num_train_workers=512, 
        num_eval_workers=16
    ) # if you are running out of ram, reduce this
    ray.get(learner.train.remote(200))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
```


I put the code into a script called **mind_games_demo.py**, so will run it via `python3 mind_games_demo.py`.
You can now open a new terminal on the same machine and track your training run via `unstable-terminal`, which will looks something like this:






TODO - this is what it should look like during training.

TODO - this is what the wandb looks like

TODO - offline evaluation

TODO - this is how you can load your checkpoint and evaluate it on textarena


TODO - here are the final results


