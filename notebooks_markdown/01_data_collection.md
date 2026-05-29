# Data Collection

Before doing any modeling, I need a reliable local copy of the NFL data. In this notebook I use `nflreadpy` to download player statistics, roster data, and schedule data for the seasons used in the project.

I save these files as raw CSVs so the later notebooks can be rerun without downloading the data every time. I am treating this notebook as the reproducible data-ingestion step, not as the place where cleaning or modeling decisions should happen.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Setup complete")
```


```python
import nflreadpy as nfl

seasons = list(range(2016, 2026))

player_stats = nfl.load_player_stats(seasons)
rosters = nfl.load_rosters(seasons)
schedules = nfl.load_schedules(seasons)

print(player_stats.shape)
print(rosters.shape)
print(schedules.shape)
```


```python
player_stats.head()

```


```python
player_stats.columns
```


```python
rosters.head()
```


```python
rosters.columns
```


```python
player_stats.write_csv("../data/raw/player_stats_2016_2025.csv")
rosters.write_csv("../data/raw/rosters_2016_2025.csv")
schedules.write_csv("../data/raw/schedules_2016_2025.csv")
```
