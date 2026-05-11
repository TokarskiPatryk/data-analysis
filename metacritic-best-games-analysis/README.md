# Metacritic Best Games Analysis

Analyse the highest-rated video games on Metacritic — both the all-time top 100 and the top 1,000 games from 2013–2022 (100 per year).

## Dataset

Scraped from Metacritic using `rvest`. Each record contains: title, platform, release date, critic score (0–100), user score (0–100). A game appears once per platform it was released on. Only games with at least 7 critic reviews are included.

Prepared data is stored as `.rds` files:
- `best_games.rds` — all-time top 100
- `best_games_2013_2022.rds` — top 1,000 (2013–2022)
- `best_games_2022.rds` — top 100 of 2022

Data scraping and cleaning is in `data-preparation-cleaning.qmd`.

## Analysis highlights

- Score distributions and critic vs user score comparison
- Platform breakdown and trends over time
- Statistical tests (e.g. score differences between platforms)

## Tech stack

R · tidyverse · rvest · rstatix · gtsummary · gt · lubridate

## Report

https://tokarskipatryk.github.io/data-analysis/metacritic-best-games-analysis/data-analysis.html
