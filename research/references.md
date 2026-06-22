# References

Annotated bibliography for Bellwether. Grouped by role. (Peer-reviewed unless marked
*preprint*; the LLM-market literature is mostly 2025–2026 preprints, re-check before citing.)

## Theory, markets as opinion pools
- **Chakraborty & Das (2015), "Market Scoring Rules Act As Opinion Pools For Risk-Averse Agents," NeurIPS.** https://proceedings.neurips.cc/paper_files/paper/2015/file/2bd7f907b7f5b6bbd91822c0c7b835f6-Paper.pdf, LMSR ≡ a logarithmic (or linear) opinion pool. *The* anchor: a market is a specific weighted average, so our question is "which pool, and when better than the mean."
- **Wolfers & Zitzewitz (2006), "Interpreting Prediction Market Prices as Probabilities," NBER WP 12200.** https://www.nber.org/system/files/working_papers/w12200/w12200.pdf, Under log utility, price = wealth-weighted mean belief.
- **Manski (2006), "Interpreting the Predictions of Prediction Markets," Economics Letters.** https://www.nber.org/system/files/working_papers/w10359/w10359.pdf, Price confounds belief with risk preferences; caveat against "price = probability."
- **Ostrovsky (2012), "Information Aggregation in Dynamic Markets with Strategic Traders," Econometrica.**, "Separable securities" fully aggregate in all equilibria; the theory benchmark Galanis uses.

## Human experiments, aggregating dispersed private information
- **Plott & Sunder (1988), "Rational Expectations and the Aggregation of Diverse Information in Laboratory Security Markets," Econometrica 56(5).** https://www.jstor.org/stable/1911360, Aggregation succeeds with a single security + identical preferences, **fails** with diverse preferences unless markets are complete. The conditional result we mirror.
- **Forsythe & Lundholm (1990), "Information Aggregation in an Experimental Market," Econometrica 58(2).** https://www.jstor.org/stable/2938205, Experience *and* common knowledge of payoffs are jointly needed for aggregation.
- **Corgnet et al. (2020), "Reconsidering Rational Expectations and the Aggregation of Diverse Information."** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3398838, Fails to replicate REE dominance under the original protocol; supports it under a prediction-market structure. Aggregation is real but fragile.

## Human forecasting, markets vs. polls/averaging
- **Atanasov et al. (2017), "Distilling the Wisdom of Crowds: Prediction Markets vs. Prediction Polls," Management Science.** https://pubsonline.informs.org/doi/10.1287/mnsc.2015.2374, Tuned, extremized, weighted polls **beat** markets. The bar averaging sets.
- **Dana, Atanasov, Tetlock & Mellers (2019), "Are markets more accurate than polls?", Judgment and Decision Making.** https://sjdm.org/~baron/journal/18/18919/jdm18919.html, Well-aggregated polls ≈ or slightly beat markets.
- **Satopää et al. (2016), "Combining multiple probability predictions using a simple logit model," / "extremizing,"** and **Baron et al. (2014), Decision Analysis**, optimal extremizing depends on information diversity; the averaging rule a market must beat.

## LLM forecasting & aggregation (by averaging)
- **Schoenegger, Tuminauskaite, Park, Tetlock (2024), "Wisdom of the Silicon Crowd," Science Advances.** https://arxiv.org/abs/2402.19379, A 12-LLM **median** ensemble rivals a 925-human crowd. The averaging baseline.
- **Halawi et al. (2024), "Approaching Human-Level Forecasting with Language Models," NeurIPS.** https://arxiv.org/abs/2402.18563, Retrieval + self-ensembling; system hedges; no market.
- **Karger et al. (2024), "ForecastBench," ICLR 2025.** https://arxiv.org/abs/2409.19839, Large, leak-free public forecasting benchmark with human/LLM baselines. *Our public benchmark.*
- **AIA Forecaster (2025).** https://arxiv.org/abs/2511.07678, Multi-agent forecaster matching superforecasters; naive LLM synthesis is *worse* than the mean, only agentic retrieval + calibration beats it. *preprint*

## LLM markets & multi-agent aggregation (the frontier)
- **Galanis (2026), "Information Aggregation with AI Agents," arXiv:2604.20050.** https://arxiv.org/abs/2604.20050, **Direct predecessor:** LLM agents trade an LMSR market on dispersed private signals; measures market-vs-truth (REE). Does *not* compare to averaging or a pooled oracle, nor vary signal structure. *preprint*
- **Li, Naito & Shirado (2025), "HiddenBench: Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs," arXiv:2505.11556.** https://arxiv.org/abs/2505.11556, LLM **debate fails** to pool hidden-profile info (≈30% vs ≈81% pooled-oracle). Our motivation for testing a *market* vs. debate. *preprint*
- **Schneider & Schramm (2025), "The Wisdom of Deliberating AI Crowds," arXiv:2512.22625.** https://arxiv.org/abs/2512.22625, Aggregation gains appear *only for diverse models* (p=0.017); median aggregation, **no market**. Validates our hypothesis shape. *preprint*
- **Kohli (2026), "Nine Judges, Two Effective Votes," arXiv:2605.29800.** https://arxiv.org/abs/2605.29800, Correlated LLM errors → low *effective* diversity. Why we measure decorrelation, not model count. *preprint*
- **Ai et al. (2025), "Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information," arXiv:2510.01499.** https://arxiv.org/abs/2510.01499, Bayesian-truth-serum-style weighting; frames optimal LLM aggregation as open. *preprint*

## Adjacent LLM-market work (to pre-empt "isn't this just X?")
- **Barot & Borkhatariya (2026), "PolySwarm," arXiv:2604.03888**, 50 LLM personas trade Polymarket (CLOB); confidence-weighted blend, not LMSR-vs-averaging. *preprint*
- **Henning et al. (2025), "LLM Agents Do Not Replicate Human Market Traders," arXiv:2502.15800**, bubble paradigm, common info; LLMs price near-rational. *preprint*
- **"Market Making as a Scalable Framework for Multi-Agent LLM Systems" (2025), arXiv:2511.17621**, bid-ask argumentation, identical models, binary QA; no market-vs-averaging baseline. *preprint*
