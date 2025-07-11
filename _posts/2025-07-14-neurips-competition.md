---
layout: post
title:  "MindGames (NeurIPS 2025 Competition)"
date:   2025-07-14
---

<!-- Optional excerpt for RSS / list views -->
The new **MindGames** competition challenges agents to learn *reasoning-heavy*
board- and card-game skills under tight compute budgets.  
In this post I walk through the tracks, show baselines, and share tips for
bootstrapping your own entry.

<!-- More tag (optional): everything above this line shows in index excerpts -->
<!--more-->

## 🗺️ Table of Contents
1. [What is MindGames?](#what-is-mindgames)
2. [Competition Tracks](#competition-tracks)
3. [Baseline Starter Kit](#baseline-starter-kit)
4. [Training Pipeline Walk-through](#training-pipeline-walk-through)
5. [Tips for a Strong Submission](#tips-for-a-strong-submission)
6. [Resources & Links](#resources--links)

---

## What is MindGames?

MindGames is a NeurIPS 2025 competition aimed at **multi-step reasoning** under
strict model-size constraints (<&nbsp;150 M parameters).  
Participants build agents that play a suite of text-based games drawn from the
*TextArena* benchmark.

## Competition Tracks

| Track | Environment | Goal | Constraint |
|-------|-------------|------|------------|
| **1** | *Liars Dice* | Maximise average reward vs. fixed bots | 50 M params |
| **2** | *Nim*        | Beat adaptive opponents | 100 M params |
| **3** | *Kuhn Poker* | Highest ELO after round-robin | 150 M params |

## Baseline Starter Kit

Clone the official repo:

```shell
git clone https://github.com/neurips-competitions/mindgames.git
cd mindgames
pip install -r requirements.txt
