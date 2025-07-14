---
layout: post
title:  "MindGames (NeurIPS 2025 Competition)"
date:   2025-07-14
---

**MindGames** is a a NeurIPS25 competition for text-based games that Bobby and I helped organize. Since we are obviously not allowed to participated, we wanted to write this short blog explaining how we would use UnstableBaselines (TODO link to it), to train a model an submit it to the competition (optimally giving you a small edge over your competition).

MindGames has two main trackes focused on TheoryOfMindGames (one for 'Secretmafia-v0' and one for `Codenames-v0`, `ColonelBlotto-v0` and `ThreePlayerIPD-v0`). In our experience, the biggest challange is training small (i.e. <8B) models on games where the action output is part of the next players observation. In this blog we will focus on track two, since `ThreePlayerIPD-v0` is the only such environment in it.

Since we have also been training on these games for a couple of months, we created version of them where the observations are structured in a training-friendly manner (you can just use the suffic `-train`; i.e. `Codenames-v0` -> `Codenames-v0-train`). This will work both for offline training and your online submission.

First things first, we need to decide on a model and build the training script. In Spiral (TODO link to paper) we used the __Qwen3-4B-Base__ model so we can show the OOD math improvement of self-play. However, in this context that won't be necessary, and since the parameter limit is 8B, we will use the __Qwen3-8B__ model (i.e. instruction tuned).

As for the training script, since 8b is rather big, we will use UnstableBaselines, which uses LoRA and activation checkpointing, thus 48GB of vRam should be enough (for this blog we will use our 3x RTX6000 ada machine; if you only have access to 24gb of vRam you will likely have to go with the 4B version).

You can install UnstableBaselines via ```pip install unstable-rl``` (this will also install TextArena).

Now we can set up the training script. I will first explain the components one by one, and then post the full script at the bottom.

First we need to import all relevant packages; specifically, **ray**, **unstable** (the package name for UnstableBaselines) and the reward transformations from UnstableBaselines.
```python3
import ray, unstable
import unstable.reward_transformations as retra
```


Now we can build initialize constants and build the two necessary configuration configs (namely, the lora_config, specifying the lora size and which layers to apply it to, and the vllm_config, specifying out generation hyperparameters):
```python3
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_GENERATION_LENGTH = 4096
MAX_TRAIN_SEQ_LEN = None # if you are running out of vRam, you can decrease this.

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

With everything imported and specified, we can now initialize ray and start building the relevant modules from unstable.
```python3
ray.init(namespace="unstable") # the namespace is mostly important for the terminal_interface.py script (which loads the modules from the "unstable" namespace)

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="Codenames-v0-train", num_players=4, num_actors=4, prompt_template="llama-instruct-zs"),
        unstable.TrainEnvSpec(env_id="ColonelBlotto-v0-train", num_players=2, num_actors=2, prompt_template="llama-instruct-zs"),
        unstable.TrainEnvSpec(env_id="ThreePlayerIPD-v0-train", num_players=3, num_actors=3, prompt_template="llama-instruct-zs"),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="Codenames-v0-train", num_players=4, prompt_template="llama-instruct-zs"),
        unstable.EvalEnvSpec(env_id="ColonelBlotto-v0-train", num_players=2, prompt_template="llama-instruct-zs"),
        unstable.EvalEnvSpec(env_id="ThreePlayerIPD-v0-train", num_players=3, prompt_template="llama-instruct-zs"),
])
```

The __TrainEnvSpec__ expects the **env_id** (same as in TextArena), the **num_players** of the environment, the **num_actors** we want to use to collect data (i.e. if num_players==num_actors we will use mirror self-play with no opponent sampling) and the **prompt_template** used to process the observations.



