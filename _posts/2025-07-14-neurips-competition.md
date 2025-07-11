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

You can install UnstableBaselines via `pip install unstable-rl` (this will also install TextArena).


