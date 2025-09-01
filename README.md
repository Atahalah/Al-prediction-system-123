# Al-prediction-system-123
The user's system 

### 1. data/__init__.py
```python name=data/__init__.py
# (empty file)
```

---

### 2. data/universal_ingest.py
```python name=data/universal_ingest.py
import os, json, logging, pandas as pd, requests
from statsbombpy import sb
from datetime import datetime, timedelta
from typing import List

logging.basicConfig(level=logging.INFO)

class UniversalIngest:
    def __init__(self, base_path: str = "bronze"):
        self.base = base_path
        os.makedirs(self.base, exist_ok=True)

    # ---------- StatsBomb ----------
    def fetch_statsbomb(self) -> None:
        matches = sb.matches(competition_id=11, season_id=1)  # EPL demo
        matches.to_parquet(f"{self.base}/statsbomb_matches.parquet")
        events = sb.events(match_id=matches["match_id"].iloc[0])
        events.to_parquet(f"{self.base}/statsbomb_events.parquet")

    # ---------- StatsBomb 360 ----------
    def fetch_statsbomb_360(self) -> None:
        matches = pd.read_parquet(f"{self.base}/statsbomb_matches.parquet")
        frames = []
        for mid in matches["match_id"]:
            try:
                f = sb.frames(match_id=mid, fmt="dataframe")
                f["match_id"] = mid
                frames.append(f)
            except Exception as e:
                logging.warning(f"360 not found for {mid}: {e}")
        pd.concat(frames).to_parquet(f"{self.base}/sb360_frames.parquet")

    # ---------- Football-Data.co.uk ----------
    def fetch_football_data(self, league: str = "E0") -> None:
        url = f"https://www.football-data.co.uk/mmz4281/{datetime.now().year}{datetime.now().year+1-2000}/{league}.csv"
        df = pd.read_csv(url)
        df.to_parquet(f"{self.base}/football_data_{league}.parquet")

    # ---------- Combine ----------
    def combine_all(self) -> pd.DataFrame:
        dfs = [pd.read_parquet(p) for p in os.listdir(self.base) if p.endswith(\".parquet\")]
        return pd.concat(dfs, ignore_index=True)

if __name__ == \"__main__\":
    ing = UniversalIngest()
    ing.fetch_statsbomb()
    ing.fetch_statsbomb_360()
    ing.fetch_football_data(\"E0\")
```

---

### 3. data/cleaners/__init__.py
```python name=data/cleaners/__init__.py
# (empty file)
```

---

### 4. data/cleaners/clean_sb360.py
```python name=data/cleaners/clean_sb360.py
import pandas as pd, numpy as np

def clean_freeze_frames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Explode freeze_frame dict column
    ff = pd.json_normalize(df["freeze_frame"])
    df = df.drop(columns=["freeze_frame"]).join(ff.add_prefix("ff_"))
    df["pressure_index"] = (
        1 / (df["ff_location"].apply(lambda x: np.linalg.norm(x) + 1e-6))
    ).groupby(df.index // 11).sum()
    return df[["match_id", "shot_id", "pressure_index"]]
```

---

### 5. data/cleaners/build_elo.py
```python name=data/cleaners/build_elo.py
import json, pandas as pd, numpy as np
from typing import Dict, List

K = 20
START_ELO = 1500

class EloTracker:
    def __init__(self):
        self.elo: Dict[int, float] = {}

    def _expected(self, a: float, b: float) -> float:
        return 1 / (1 + 10 ** ((b - a) / 400))

    def update_match(self, home_players: List[int], away_players: List[int],
                     home_goals: int, away_goals: int) -> None:
        home_avg = np.mean([self.elo.get(p, START_ELO) for p in home_players])
        away_avg = np.mean([self.elo.get(p, START_ELO) for p in away_players])
        result = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
        for p in home_players:
            e = self._expected(home_avg, away_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (result - e)
        for p in away_players:
            e = self._expected(away_avg, home_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (1 - result - (1 - e))

def build_elo_df(matches_df: pd.DataFrame) -> Dict[int, float]:
    tracker = EloTracker()
    for _, row in matches_df.iterrows():
        tracker.update_match(
            json.loads(row["home_lineup"]),
            json.loads(row["away_lineup"]),
            row["home_goals"],
            row["away_goals"],
        )
    return tracker.elo
```

---
Thank you for confirming! Here are the files and their contents for the first batch. You (or I, if I could write directly) should create these files in your repo Atahalah/Al-prediction-system-123.

Please copy-paste the following into your project in the correct paths, or let me know if you want to proceed file-by-file. After this batch, I will continue with the next.

---

```python name=data/__init__.py
# empty
```

```python name=data/universal_ingest.py
import os, json, logging, pandas as pd, requests
from statsbombpy import sb
from datetime import datetime, timedelta
from typing import List

logging.basicConfig(level=logging.INFO)

class UniversalIngest:
    def __init__(self, base_path: str = "bronze"):
        self.base = base_path
        os.makedirs(self.base, exist_ok=True)

    # ---------- StatsBomb ----------
    def fetch_statsbomb(self) -> None:
        matches = sb.matches(competition_id=11, season_id=1)  # EPL demo
        matches.to_parquet(f"{self.base}/statsbomb_matches.parquet")
        events = sb.events(match_id=matches["match_id"].iloc[0])
        events.to_parquet(f"{self.base}/statsbomb_events.parquet")

    # ---------- StatsBomb 360 ----------
    def fetch_statsbomb_360(self) -> None:
        matches = pd.read_parquet(f"{self.base}/statsbomb_matches.parquet")
        frames = []
        for mid in matches["match_id"]:
            try:
                f = sb.frames(match_id=mid, fmt="dataframe")
                f["match_id"] = mid
                frames.append(f)
            except Exception as e:
                logging.warning(f"360 not found for {mid}: {e}")
        pd.concat(frames).to_parquet(f"{self.base}/sb360_frames.parquet")

    # ---------- Football-Data.co.uk ----------
    def fetch_football_data(self, league: str = "E0") -> None:
        url = f"https://www.football-data.co.uk/mmz4281/{datetime.now().year}{datetime.now().year+1-2000}/{league}.csv"
        df = pd.read_csv(url)
        df.to_parquet(f"{self.base}/football_data_{league}.parquet")

    # ---------- Combine ----------
    def combine_all(self) -> pd.DataFrame:
        dfs = [pd.read_parquet(p) for p in os.listdir(self.base) if p.endswith(".parquet")]
        return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    ing = UniversalIngest()
    ing.fetch_statsbomb()
    ing.fetch_statsbomb_360()
    ing.fetch_football_data("E0")
```

```python name=data/cleaners/__init__.py
# empty
```

```python name=data/cleaners/clean_sb360.py
import pandas as pd, numpy as np

def clean_freeze_frames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Explode freeze_frame dict column
    ff = pd.json_normalize(df["freeze_frame"])
    df = df.drop(columns=["freeze_frame"]).join(ff.add_prefix("ff_"))
    df["pressure_index"] = (
        1 / (df["ff_location"].apply(lambda x: np.linalg.norm(x) + 1e-6))
    ).groupby(df.index // 11).sum()
    return df[["match_id", "shot_id", "pressure_index"]]
```

```python name=data/cleaners/build_elo.py
import json, pandas as pd, numpy as np
from typing import Dict, List

K = 20
START_ELO = 1500

class EloTracker:
    def __init__(self):
        self.elo: Dict[int, float] = {}

    def _expected(self, a: float, b: float) -> float:
        return 1 / (1 + 10 ** ((b - a) / 400))

    def update_match(self, home_players: List[int], away_players: List[int],
                     home_goals: int, away_goals: int) -> None:
        home_avg = np.mean([self.elo.get(p, START_ELO) for p in home_players])
        away_avg = np.mean([self.elo.get(p, START_ELO) for p in away_players])
        result = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
        for p in home_players:
            e = self._expected(home_avg, away_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (result - e)
        for p in away_players:
            e = self._expected(away_avg, home_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (1 - result - (1 - e))

def build_elo_df(matches_df: pd.DataFrame) -> Dict[int, float]:
    tracker = EloTracker()
    for _, row in matches_df.iterrows():
        tracker.update_match(
            json.loads(row["home_lineup"]),
            json.loads(row["away_lineup"]),
            row["home_goals"],
            row["away_goals"],
        )
    return tracker.elo
```

---

```python name=data/__init__.py
# empty
```

```python name=data/universal_ingest.py
import os, json, logging, pandas as pd, requests
from statsbombpy import sb
from datetime import datetime, timedelta
from typing import List

logging.basicConfig(level=logging.INFO)

class UniversalIngest:
    def __init__(self, base_path: str = "bronze"):
        self.base = base_path
        os.makedirs(self.base, exist_ok=True)

    # ---------- StatsBomb ----------
    def fetch_statsbomb(self) -> None:
        matches = sb.matches(competition_id=11, season_id=1)  # EPL demo
        matches.to_parquet(f"{self.base}/statsbomb_matches.parquet")
        events = sb.events(match_id=matches["match_id"].iloc[0])
        events.to_parquet(f"{self.base}/statsbomb_events.parquet")

    # ---------- StatsBomb 360 ----------
    def fetch_statsbomb_360(self) -> None:
        matches = pd.read_parquet(f"{self.base}/statsbomb_matches.parquet")
        frames = []
        for mid in matches["match_id"]:
            try:
                f = sb.frames(match_id=mid, fmt="dataframe")
                f["match_id"] = mid
                frames.append(f)
            except Exception as e:
                logging.warning(f"360 not found for {mid}: {e}")
        pd.concat(frames).to_parquet(f"{self.base}/sb360_frames.parquet")

    # ---------- Football-Data.co.uk ----------
    def fetch_football_data(self, league: str = "E0") -> None:
        url = f"https://www.football-data.co.uk/mmz4281/{datetime.now().year}{datetime.now().year+1-2000}/{league}.csv"
        df = pd.read_csv(url)
        df.to_parquet(f"{self.base}/football_data_{league}.parquet")

    # ---------- Combine ----------
    def combine_all(self) -> pd.DataFrame:
        dfs = [pd.read_parquet(p) for p in os.listdir(self.base) if p.endswith(".parquet")]
        return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    ing = UniversalIngest()
    ing.fetch_statsbomb()
    ing.fetch_statsbomb_360()
    ing.fetch_football_data("E0")
```

```python name=data/cleaners/__init__.py
# empty
```

```python name=data/cleaners/clean_sb360.py
import pandas as pd, numpy as np

def clean_freeze_frames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Explode freeze_frame dict column
    ff = pd.json_normalize(df["freeze_frame"])
    df = df.drop(columns=["freeze_frame"]).join(ff.add_prefix("ff_"))
    df["pressure_index"] = (
        1 / (df["ff_location"].apply(lambda x: np.linalg.norm(x) + 1e-6))
    ).groupby(df.index // 11).sum()
    return df[["match_id", "shot_id", "pressure_index"]]
```

```python name=data/cleaners/build_elo.py
import json, pandas as pd, numpy as np
from typing import Dict, List

K = 20
START_ELO = 1500

class EloTracker:
    def __init__(self):
        self.elo: Dict[int, float] = {}

    def _expected(self, a: float, b: float) -> float:
        return 1 / (1 + 10 ** ((b - a) / 400))

    def update_match(self, home_players: List[int], away_players: List[int],
                     home_goals: int, away_goals: int) -> None:
        home_avg = np.mean([self.elo.get(p, START_ELO) for p in home_players])
        away_avg = np.mean([self.elo.get(p, START_ELO) for p in away_players])
        result = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0
        for p in home_players:
            e = self._expected(home_avg, away_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (result - e)
        for p in away_players:
            e = self._expected(away_avg, home_avg)
            self.elo[p] = self.elo.get(p, START_ELO) + K * (1 - result - (1 - e))

def build_elo_df(matches_df: pd.DataFrame) -> Dict[int, float]:
    tracker = EloTracker()
    for _, row in matches_df.iterrows():
        tracker.update_match(
            json.loads(row["home_lineup"]),
            json.loads(row["away_lineup"]),
            row["home_goals"],
            row["away_goals"],
        )
    return tracker.elo
```

---

---

```python name=processing/__init__.py
# empty
```

```python name=processing/gold_features.py
import pandas as pd, numpy as np
from typing import Dict

def add_time_decay(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_decay_weight"] = 1 / (1 + (pd.Timestamp("now") - pd.to_datetime(df["date"])).dt.days / 365)
    return df

def add_market_implied(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_odds", "draw_odds", "away_odds"]:
        df[f"market_{col}"] = 1 / df[col]
    return df

def add_rolling_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = df.sort_values(["team", "date"])
    df[f"form_last{n}"] = df.groupby("team")["goals"].transform(lambda x: x.rolling(n).mean())
    return df

def team_elo_features(match: pd.Series, player_elo: Dict[int, float]) -> Dict[str, float]:
    home = match["home_players"]
    away = match["away_players"]
    home_elo = np.mean([player_elo.get(p, 1500) for p in home])
    away_elo = np.mean([player_elo.get(p, 1500) for p in away])
    return {"home_avg_elo": home_elo, "away_avg_elo": away_elo, "elo_gap": home_elo - away_elo}
```

```python name=processing/stack_dataset.py
import pandas as pd, numpy as np

def build_stack_df(gold_df: pd.DataFrame) -> (pd.DataFrame, np.ndarray):
    models = ["rf", "lgb", "graphformer"]
    probs = [pd.read_parquet(f"preds/{m}_val.parquet") for m in models]
    X = np.hstack(probs)
    y = gold_df["outcome"].values
    cols = [f"{m}_{cls}" for m in models for cls in ["H", "D", "A"]]
    return pd.DataFrame(X, columns=cols), y
```

```python name=models/__init__.py
# empty
```

```python name=models/graphformer.py
import torch, torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import Encoder

class GraphFormer(nn.Module):
    def __init__(self, tab_dim: int, hidden: int = 128, heads: int = 4):
        super().__init__()
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.transformer = Encoder(
            d_model=hidden, nhead=heads, num_layers=2
        )
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.fusion = nn.Linear(3 * hidden, hidden)
        self.pi = nn.Linear(hidden, 3)
        self.mu = nn.Linear(hidden, 6)
        self.logvar = nn.Linear(hidden, 6)

    def forward(self, graph, seq, tab):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.conv2(self.conv1(x, edge_index)).relu()
        g_emb = global_mean_pool(x, batch)
        seq_emb = self.transformer(seq).mean(dim=1)
        tab_emb = self.tab_mlp(tab)
        fused = torch.cat([g_emb, seq_emb, tab_emb], dim=1)
        return self.pi(self.fusion(fused)), self.mu(self.fusion(fused)), self.logvar(self.fusion(fused))
```

```python name=models/bayesian_stack.py
import pymc as pm, aesara.tensor as at
import arviz as az, numpy as np

def fit_bayesian_stack(X: np.ndarray, y: np.ndarray, chains: int = 2, draws: int = 2000):
    n_obs, n_feat = X.shape
    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.ones(n_feat))
        intercept = pm.Normal("intercept", 0, 1)
        logits = intercept + at.dot(X, w)
        outcome = pm.Categorical("outcome", p=pm.math.softmax(logits), observed=y)
        idata = pm.sample(chains=chains, draws=draws, target_accept=0.9)
    return idata
```

```python name=models/online_model.py
from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics

clf = AdaptiveRandomForestClassifier(n_models=50, max_depth=12, grace_period=50, seed=42)
metric = metrics.LogLoss()
```

```python name=training/__init__.py
# empty
```

```python name=training/global_trainer_20k.py
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

gold = pd.read_parquet("gold/gold.parquet")
X = gold.drop(columns=["outcome"])
y = gold["outcome"]

rf = RandomForestClassifier(n_estimators=500).fit(X, y)
lgb = LGBMClassifier(n_estimators=500).fit(X, y)

joblib.dump(rf, "models/rf.joblib")
joblib.dump(lgb, "models/lgb.joblib")
```

```python name=training/train_graphformer.py
import optuna, torch, joblib
from models.graphformer import GraphFormer
from dataloaders.graphformer_dataset import MatchDataset

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden = trial.suggest_int("hidden", 64, 256, step=32)
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    # … full training loop …
    return best_val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

```python name=training/retrain_online.py
import joblib, time, redis, pandas as pd
from models.online_model import clf, metric
from monitoring.drift_detector import detect_drift

r = redis.Redis(host="redis", port=6379, decode_responses=True)
ref = pd.read_parquet("gold/reference.parquet")

while True:
    latest = pd.read_parquet("gold/live_window.parquet")
    if detect_drift(latest, ref):
        for _, row in latest.iterrows():
            x = row.drop("outcome").to_dict()
            y = row["outcome"]
            proba = clf.predict_proba_one(x)
            metric.update(y, proba)
            clf.learn_one(x, y)
        joblib.dump(clf, "models/online_rf.joblib")
        r.set("model_version", "online_rf")
    time.sleep(3600)
```

---
```python name=api/__init__.py
# empty
```

```python name=api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.predictor import get_predictor

app = FastAPI(title="Free Football Predictor", version="1.0")

class PredictionRequest(BaseModel):
    home: str
    away: str
    kickoff: str

class PredictionResponse(BaseModel):
    home: float
    draw: float
    away: float

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict_match(req: PredictionRequest):
    predictor = get_predictor()
    probs = predictor.predict(req.home, req.away, req.kickoff)
    return PredictionResponse(home=probs[0], draw=probs[1], away=probs[2])
```

```python name=api/predictor.py
import joblib, redis, json, numpy as np, pymc as pm
from models.online_model import clf

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

class BayesianStackPredictor:
    def __init__(self, path="models/bayesian_stack_post.nc"):
        import arviz as az
        self.idata = az.from_netcdf(path)
        self.w = self.idata.posterior["w"].mean(dim=("chain", "draw")).values
        self.intercept = self.idata.posterior["intercept"].mean(dim=("chain", "draw")).values

    def predict(self, rf, lgb, graph):
        X = np.concatenate([rf, lgb, graph])
        logits = self.intercept + X @ self.w.T
        return pm.math.softmax(logits).eval()

class OnlinePredictor:
    def predict(self, *args):
        # dummy feature extraction
        x = {"dummy": 0.5}
        proba = clf.predict_proba_one(x)
        return [proba.get(0, 0), proba.get(1, 0), proba.get(2, 0)]

def get_predictor():
    version = redis_client.get("model_version")
    if version == "online_rf":
        return OnlinePredictor()
    return BayesianStackPredictor()
```

```python name=dashboard/__init__.py
# empty
```

```python name=dashboard/app.py
import streamlit as st, requests
st.set_page_config(page_title="Free Football Predictions", layout="wide")
st.title("🌍 Global Free Football Predictions")

home = st.text_input("Home Team")
away = st.text_input("Away Team")
kickoff = st.date_input("Kick-off Date")

if st.button("Predict"):
    r = requests.post("http://api:8000/predict", json={"home": home, "away": away, "kickoff": str(kickoff)})
    if r.ok:
        st.json(r.json())
    else:
        st.error("API unreachable")
```

```python name=monitoring/__init__.py
# empty
```

```python name=monitoring/drift_detector.py
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd

def detect_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame, threshold: float = 0.05) -> bool:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    score = report.as_dict()["metrics"][0]["result"]["dataset_drift"]["score"]
    return score > threshold
```

```python name=tests/__init__.py
# empty
```

```python name=tests/integration/__init__.py
# empty
```

```python name=tests/integration/test_full_pipeline.py
# (add your integration test code here)
```

```python name=tests/unit/__init__.py
# empty
```

```python name=tests/unit/test_sb360.py
# (add your unit test for sb360 here)
```

```python name=tests/unit/test_elo.py
# (add your unit test for elo here)
```

```python name=tests/unit/test_graphformer.py
# (add your unit test for graphformer here)
```

```python name=tests/unit/test_bayesian_stack.py
# (add your unit test for bayesian_stack here)
```

```text name=models/.gitkeep
# empty file to keep the models directory tracked in git
```
Dockerfile.retrain


