# ML System Design — aggregated references

Curated reading list for the ML system design topic. Each drill carries its own narrow references; this file is the broader background.

## Foundational

- [Sculley et al. — "Hidden Technical Debt in Machine Learning Systems"](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — the canonical paper on the costs of ML in production.
- [Huyen — *Designing Machine Learning Systems* (book)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — comprehensive textbook.
- [Chip Huyen — *machine-learning-systems-design* (open booklet)](https://github.com/chiphuyen/machine-learning-systems-design) — 27 open-ended MLSD questions.

## Recsys / Search / Ranking

- [Covington, Adomavicius, Sargin — "Deep Neural Networks for YouTube Recommendations"](https://research.google/pubs/pub45530/) — two-stage funnel.
- [Naumov et al. — "DLRM"](https://arxiv.org/abs/1906.00091) — modern recsys architecture.
- [Cheng et al. — "Wide & Deep Learning for Recommender Systems"](https://arxiv.org/abs/1606.07792) — Wide & Deep.
- [Burges — "From RankNet to LambdaRank to LambdaMART"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) — LTR.
- [Robertson & Zaragoza — BM25 reference](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf) — BM25.

## CTR / Ad ranking

- [McMahan et al. — "Ad Click Prediction: a View from the Trenches"](https://research.google/pubs/pub41159/) — FTRL & production lessons.
- [Guo et al. — "DeepFM"](https://arxiv.org/abs/1703.04247) — DeepFM.
- [Wang et al. — "DCN V2"](https://arxiv.org/abs/2008.13535) — DCN-V2.

## LLM-app system design

- [Lewis et al. — "RAG"](https://arxiv.org/abs/2005.11401) — RAG paradigm.
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents) — single-agent + tools pattern.
- [Yao et al. — "ReAct"](https://arxiv.org/abs/2210.03629) — reasoning + acting.
- [Jimenez et al. — "SWE-bench"](https://arxiv.org/abs/2310.06770) — coding-agent eval.

## Feature store / platform

- [Tecton — Feature Store architecture](https://docs.tecton.ai/) — primary docs.
- [Feast project](https://feast.dev/) — open-source feature store.
- [Uber — Michelangelo platform overview](https://www.uber.com/blog/michelangelo-machine-learning-platform/) — production case study.

## Adversarial / safety

- [Microsoft PhotoDNA](https://www.microsoft.com/en-us/photodna) — CSAM hash matching.
- [Mazeika et al. — "HarmBench"](https://arxiv.org/abs/2402.04249) — adversarial moderation eval.

## Operations / monitoring

- [Google — "Rules of Machine Learning"](https://developers.google.com/machine-learning/guides/rules-of-ml) — practical ML-in-production rules.

## Books (full-length references)

- Huyen, *Designing Machine Learning Systems* (O'Reilly).
- Lakshmanan, Robinson, Munn, *Machine Learning Design Patterns* (O'Reilly).
- Kleppmann, *Designing Data-Intensive Applications* — not ML-specific but essential for the systems half.
