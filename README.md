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
