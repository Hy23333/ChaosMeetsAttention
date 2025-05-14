# Chaos Meets Attention: Transformers for Large-Scale Dynamical Prediction (ICML 2025 Poster)

Thanks for your attention in our new work on predicting large scale chaos (ergodic).

[Paper link](https://doi.org/10.48550/arXiv.2504.20858)

Generating long-term trajectories of dissipative chaotic systems autoregressively is a highly challenging task. The inherent positive Lyapunov exponents amplify prediction errors over time.
Many chaotic systems possess a crucial property — ergodicity on their attractors, which makes long-term prediction possible. 
State-of-the-art methods address ergodicity by preserving statistical properties using optimal transport techniques. However, these methods face scalability challenges due to the curse of dimensionality when matching distributions. To overcome this bottleneck, we propose a scalable transformer-based framework capable of stably generating long-term high-dimensional and high-resolution chaotic dynamics while preserving ergodicity. Our method is grounded in a physical perspective, revisiting the Von Neumann mean ergodic theorem to ensure the preservation of long-term statistics in the $\mathcal{L}^2$ space. We introduce novel modifications to the attention mechanism, making the transformer architecture well-suited for learning large-scale chaotic systems. Compared to operator-based and transformer-based methods, our model achieves better performances across five metrics, from short-term prediction accuracy to long-term statistics. In addition to our methodological contributions, we introduce a new chaotic system benchmark: a machine learning dataset of 140$k$ snapshots of turbulent channel flow along with various evaluation metrics for both short- and long-term performances, which is well-suited for machine learning research on chaotic systems.

# Benchmark datasets 

Data generation details are included in the paper Appendix D and G.

~~The dataset sample of Turbulent Channel Flow is available via the link (https://filebin.net/37p4dxup0t320143)~~

The benchmark dataset will be available soon via Zenodo.

# Others: Supplementay code submission to ICML 2025

The code in supplementary material is developed for modeling large-scale chaos, modifying the base code from the repo [FactFormer](https://github.com/BaratiLab/FactFormer).

Please incorporate datasets into the `/data` folder and define your local path before you start training or evaluation.

We suggest you try with requesting a GPU card memory larger than 24GB when training.
