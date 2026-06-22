# Landscape, who does what (mid-2026)

The space splits into five buckets. None covers the full "internal + create event + AI-agent traders +
price = probability" loop.

## 1. AI agents trading on *human* public markets
Bots trading alongside humans on Polymarket / Kalshi. Not agent-only, not internal.
- AI agents are ~30%+ of Polymarket wallet activity; 14 of top-20 profitable wallets are bots.
- Polystrat, OpenClaw, IronClaw, autonomous trader agents.
- Source: [CoinDesk, AI agents are quietly rewriting prediction market trading](https://www.coindesk.com/tech/2026/03/15/ai-agents-are-quietly-rewriting-prediction-market-trading)

## 2. Agent-only forecasting arenas / benchmarks
Only AI models compete, but these are scored competitions, not markets with internal price discovery.
- **Prophet Arena**, continuous probabilistic forecasts on real events; scores calibration + market profitability. [arXiv 2510.17638](https://arxiv.org/abs/2510.17638)
- **Prediction Arena**, each model gets ~$10k to trade live on Kalshi/Polymarket over ~57 days. [arXiv 2604.07355](https://arxiv.org/abs/2604.07355)
- **PrediBench** (PresageLabs), agents bet $1 on top-10 Polymarket questions; Brier + PnL. [HF blog](https://huggingface.co/blog/charles-azam/predibench)
- **FutureBench** (Together AI) / **FutureX** (ByteDance/Fudan), agent-only forecasting benchmarks from scraped news. [HF blog](https://huggingface.co/blog/futurebench)
- **Metaculus AI tournaments** + open-source bot framework. [forecasting-tools](https://github.com/Metaculus/forecasting-tools)

## 3. Agent-driven markets (public / crypto)
Closest to "agents form the market," but public and on-chain, not internal.
- **Olas Predict (Omen on Gnosis)**, agent roles: Market Creators (deploy+fund), Traders, Prediction Brokers/Mechs (probabilities), Closers (resolve). Humans excluded. Run agents no-code via Pearl. [Olas Predict](https://olas.network/agent-economies/predict)
- **Rain**, OpenClaw-based SDK: single prompt → live market (deploy, oracle, liquidity, quoting agent). Agent-ready, not agent-only; humans can trade. $5M grant program. [Crypto Briefing](https://cryptobriefing.com/rain-launches-an-openclaw-and-ai-agent-ready-sdk-for-building-independent-prediction-market-platforms-and-a-5m-grant-program/)

## 4. Enterprise / internal forecasting platforms (built for humans)
The internal, self-hostable base, but human forecasters, agents only via API.
- **Cultivate Labs (Cultivate Forecasts)**, self-host via Docker, REST API, real prediction markets + simple forecasting. Separate "AI Forecaster" tool. [Hosting/features](https://www.cultivatelabs.com/forecasts_features_hosting)
- **Metaculus Private Instances**, managed private platform for org questions; aggregation not market. [Private instances](https://www.metaculus.com/services/private-instances/)
- Others historically: Hypermind, Almanis, Good Judgment.

## 5. Agent-swarm simulation for enterprise decisions
- **MiroFish**, upload internal + market data, define stakeholders, run multi-agent simulation of how a merger / launch / supply-chain decision ripples out. Simulation, not a market (no order book / price). [MiroFish](https://mirofish.work/)

## Takeaway
Build = (internal market engine with price discovery) + (diverse AI agent traders via API) +
(internal resolution oracle from Jira/CRM/finance). Cultivate gives the best market base; Metaculus the
best ready-made bots; Olas/Rain the best agent-market patterns to copy.
