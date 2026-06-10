# Anime Hit Anatomy 🎬

**What turns an anime into a hit?** We dug into ~12,000 titles to find out.

This project analyzes the structural attributes that influence anime ratings — format, episode count, genre, and popularity — and uses unsupervised learning to uncover four distinct "anime archetypes" hiding in the data.

📖 **Read the full story:** [What Turns an Anime Into a Hit? We Dug Into 12,000 Titles to Find Out](https://medium.com/@shanzay.omar/what-turns-an-anime-into-a-hit-we-dug-into-12-000-titles-to-find-out-35891f0c1abb) (Medium)
Article by [Shanzay Omar](https://medium.com/@shanzay.omar).
---

## Research Question

> What structural attributes significantly influence anime ratings, and can we identify meaningful anime archetypes through clustering?

## Dataset

[Anime Recommendation Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) (Kaggle) — metadata for ~12,000 anime titles including genre, format type, episode count, community rating, and member count (a proxy for popularity).

To reproduce: download `anime.csv` and `rating.csv` from Kaggle and place them in the repo root (or your notebook working directory).

## Methods

| Stage | Technique |
|---|---|
| Data cleaning | Missing-value handling, type conversion, genre parsing (multi-label explode) |
| Exploratory analysis | Rating distributions, genre/type/episode-length comparisons, popularity correlation |
| Hypothesis testing | Welch's t-test (TV vs. Movie ratings) + pairwise comparisons with Bonferroni correction |
| Predictive modeling | Multiple linear regression with manually computed coefficient p-values, residual diagnostics |
| Unsupervised learning | K-Means clustering (K = 4, validated via elbow method) on standardized rating, log-popularity, and episode count |

## Key Findings

**1. TV beats Movies — significantly.**
TV series have the highest average rating of any format (6.90), and a Welch's t-test confirms they significantly outperform Movies. This challenges the intuition that standalone, polished films would rate higher than serialized shows.

**2. Popularity is the strongest predictor of rating.**
Log-transformed member count is the most statistically significant predictor in the regression by a wide margin (r ≈ 0.38 in EDA). Several genre and type effects that looked strong in EDA disappear after controlling for popularity — they were confounded, not causal.

**3. There's an episode-count sweet spot.**
Single-episode titles score lowest on average (~6.22), while mid-length series (13–50 episodes) hit a storytelling sweet spot (~6.95+). The relationship is non-linear: short content may not build enough audience investment.

**4. Four anime archetypes emerge from clustering:**

| Archetype | Profile |
|---|---|
| 🏆 **Mainstream Hits** | High rating, huge audience — the titles that dominate fan culture |
| 💎 **Hidden Gems** | Strong ratings, small communities — quality that hasn't broken through |
| ⚡ **Popular but Divisive** | Large audiences, lower ratings — mass appeal meets critical fans |
| 🔬 **Niche & Low-Rated** | Small audiences, modest ratings — experimental or narrow-taste content |

The archetype structure confirms that **quality and exposure are genuinely separable dimensions** — rating and popularity are only partially correlated.

## Repository Structure

```
anime-hit-anatomy/
├── anime_hit_analysis.ipynb   # Full analysis pipeline (EDA → t-test → regression → clustering)
└── README.md
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `Matplotlib` · `SciPy` · `scikit-learn`
