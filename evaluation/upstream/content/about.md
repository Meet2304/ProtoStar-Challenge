## The challenge

**Climate Risk Intelligence: AI systems to analyze extreme weather trends and recommend climate adaptation strategies.**

### The Urban Climate AI Challenge: The 2019–2025 Test

A major Carnegie Mellon climate study (Lai & Dzombak) collected weather records for **210 U.S. cities** from ~1900 up through 2018.

Your job is simple: put yourself at the end of 2018 and **predict the actual yearly weather for each city from 2019 through 2025**, without using any data that reveals how those years turned out. You are welcome to bring in **other data** too (e.g., weather records up to 2018 from other sources, census, land cover, building stock); the data rule below says what can feed your forecasts. We will compare your predictions against the weather actually recorded at airport and city weather stations during those years. Then, use what you learned to propose one **policy idea** that helps a city, or the country, cope with a climate risk.

> **Data rule (honor code):** to forecast **temperature, precipitation, hottest day and CDD**, use only records up to **2018-12-31**: the challenge dataset, or other sources cut at the same date. Do not use **any 2019–2025 data that responds to the weather** as an input to your forecasts: weather observations from any source, and anything driven by them (e.g., streamflow, air quality, energy demand, plant or disease data). This includes numbers produced by AI models that learned from later data (see the FAQ). Other data (e.g., census, land cover, energy, health) is welcome for your analysis and policy idea; list every external source in your README. Examples are in the FAQ (*Data & submission* tab).

**Important dates** (all times ET):

- **Tuesday, November 10, 11:59 PM:** online submission deadline; registration stays open until then.
- **November 11–12:** finalist teams announced.
- **Friday, November 13:** overall winners announced.
- <span class="cmu-tag">For CMU teams</span>
  - **Tuesday, November 3, 11:59 PM:** registration deadline.
  - **Wednesday, November 4, 11 AM–12 PM:** virtual kickoff on Zoom: challenge overview, resources and Q&A.
  - **Friday, November 13:** in-person presentations of the CMU finalists, where the winning CMU team will be chosen, Hamburg Hall, Room A301, Carnegie Mellon University.

### Important updates

**November 6:** clarifications only; the rules, the scoring, the answer key and the deadline are unchanged.

- Rows not scored: the 10-day rule also applies to the 30 training years of the baselines, and a city–metric needs at least 10 valid years in 1989–2018 to have a baseline (9 city–metrics, 63 rows).
- Practice run: with the official 10-year minimum, the CDD baselines score 212 / 175 °F-days (213 / 177 were computed without it; thanks to Team Zhoey for catching it). The difference is within noise.
- Data rule: choosing or tuning a model by scoring it on 2019–2025 observations counts as using them (FAQ, *Data & submission* tab).

**November 5:** the rules, the scoring and the deadline are unchanged; `daily` and `annual_indices` are unchanged.

- New subset `coverage_and_stations`: for each city and year, the days with both temperatures and the station the record comes from.
- The dataset page now explains which days each value uses, and how missing days and station changes affect the record.
- `city_info`: station start dates are stored as dates (add `.astype(str)` if your code reads them as text); two station-date typos corrected.
- Submission template: `code/requirements.txt` added.

## Teams

Registration is team-based. Each participant can be in one team only. **A maximum of 5 individuals per team.**

**CMU team:** a team counts as a CMU team, and can win the CMU prize, when **more than half of its members are current Carnegie Mellon University students** (e.g., 2 of 3, 3 of 4, or 3 of 5). CMU students must register with their CMU email (`@andrew.cmu.edu`). Team membership at the deadline counts.

## What your model predicts

For each of the 210 cities, predict a single number for each year from 2019 through 2025 for four metrics:

| Metric | `metric` code | Unit | Description |
|---|---|---|---|
| **Average Annual Temperature** | `annual_avg_temp` | °F | The overall yearly average temperature. |
| **Total Annual Precipitation** | `total_precip_in` | inches | Total liquid moisture (rain plus melted snow) recorded for the year. |
| **Hottest Day of the Year** | `hottest_day_temp` | °F | The single highest daily peak temperature recorded that year. |
| **Cooling Degree Days (CDD)** | `cdd` | °F-days | A single yearly number that adds up how many degrees each day's average temperature rose above 65 °F; used to estimate how much air conditioning a city needs. |

### How CDD is computed

$$
\text{CDD} = \sum_{d=1}^{D} \max\left(0,\ \frac{T_{\max,d} + T_{\min,d}}{2} - 65\right)
$$

- **D:** the number of days in the year (365 or 366).
- **T<sub>max,d</sub> and T<sub>min,d</sub>:** that day's highest and lowest temperatures, in °F.
- **max(0, …):** a day at or below 65 °F adds nothing.

**Example:** a day with a high of 90 °F and a low of 70 °F averages 80 °F, so it adds 80 − 65 = 15 to the year's CDD.

No probabilities or intervals required: just your model's best guess for each value.

## How the forecasts are scored

Each metric is scored with **Mean Absolute Error (MAE)**, the average gap between your prediction and what actually happened (lower is better):

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \text{Predicted}_i - \text{Actual}_i \right|
$$

The four metrics use different units, so each one is compared against the 30-year-average baseline. The **Score** says how much better than that baseline your model was:

$$
S_m = 1 - \frac{\text{MAE}_{m,\ \text{your model}}}{\text{MAE}_{m,\ \text{30-year average}}}
\qquad
\text{Score} = 100 \times \max\left(0,\ \frac{S_{\text{temp}} + S_{\text{precip}} + S_{\text{hot}} + S_{\text{cdd}}}{4}\right)
$$

**Score = 0:** no better than the 30-year average. **Score = 100:** a perfect forecast. Higher is better.

| Score | Years scored |
|---|---|
| **Live Score** | 2019–2021 |
| **Final Score** | 2022–2025 |

The two are kept separate on purpose: a model tuned to 2019–2021 does not necessarily predict later years well.

> **Note:** a few rows are not scored: years where the station missed more than 10 days of the values a metric uses (daily maximum and minimum temperature for average temperature, hottest day and CDD; daily precipitation for total precipitation), and city–metrics with fewer than 10 valid years in 1989–2018 (the same 10-day rule), which have no baseline: 9 city–metrics, 63 rows. Predict every row anyway.

## Live leaderboard

Your **Live Score** (2019–2021) appears on the leaderboard **a few hours** after you submit. Use it to **check how your model is doing** as you improve it; the final ranking and the winners are decided after the deadline (see *Judging*). If you have questions, post in the Space's [Community tab](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/discussions).

## What to beat

The leaderboard includes three simple reference baselines:

1. **Last Year Repeated:** every year from 2019 to 2025 repeats the city's last valid value (2018 for most cities).
2. **30-Year Average:** every year equals the city's average from 1989 to 2018.
3. **Linear Trend:** a straight line through the last 30 years, extended out to 2025.

> **Target:** if your model cannot beat the 30-year average, it is probably overfitting to historical noise. Hottest day and precipitation are noisy: matching the 30-year average on them is already reasonable; most of the gain is in temperature and CDD.

> **CDD target:** beat the 30-year average (level 1), then the linear trend (level 2). In a practice run (train up to 2010, predict 2011–2018), the 30-year average scores an MAE of **212 °F-days** and the trend line **175 °F-days**, using the official rule of at least 10 valid years in the 30-year window. Thanks to **Team Zhoey** for spotting that the earlier figures (213 / 177) were computed without that rule. CDD differences under **10 °F-days** are within year-to-year noise.

## Policy idea

Write a short *Policy Idea* section (half a page at most) in your submission's README.

**The task, in one sentence:** using what you learned from the data, suggest one idea that could help a city, or the country, cope with a climate risk sustainably. Answer these three questions however you like:

1. **What risk did you notice?** E.g., more cooling days, hotter peak days, wetter or drier years.
2. **What would you do about it, and who would do it?** A city, a state, the federal government, a utility, a school, a business: your choice.
3. **Why is it sustainable?** It helps without making emissions or costs worse in the long run.

Any city, group of cities or the whole U.S.; any risk in the data; bold ideas welcome. Inspiration: [EPA Heat Island Reduction](https://www.epa.gov/heatislands), [heat.gov](https://www.heat.gov).

## What you submit

**A link to your Hugging Face dataset**, containing your forecasts (`submission.csv`), a README with your method, the AI tools you used, policy idea and feedback to the ecosystem, your code (`code/`), and the curated data it uses (`data/`), so anyone can reproduce your forecasts. **Which data you can curate, and how to use AI agents, is covered in the FAQ (*Data & submission* tab).**

You can revise and submit **up to 10 times a day** until the deadline. **Your latest valid submission counts.** Details are in the *Data & submission* tab.

## Judging

After the submission deadline, a panel of 3–4 judges selects the **finalist teams** (announced November 11–12). The **overall winners** are announced on **Friday, November 13**.

> **For CMU teams:** CMU finalists present their work in person on Friday, November 13 (Hamburg Hall, Room A301, Carnegie Mellon University), where the winning CMU team will be chosen.

Judges use six criteria:

<!-- RUBRIC_TABLE -->

Judges consider your **Score** and method under *Scientific, Research, or Societal Value* and *Technical Creativity*, and your **policy idea** under *Scientific, Research, or Societal Value*. **Together these two criteria carry 20% of the rubric's weight: the Score is evidence the judges read, not points added to their evaluation.**

Write your **Feedback to the Ecosystem** in its own section of your README: what got in the way (data, metadata, documentation, licensing, tools) and what would fix it.

If your main finding is that reuse or reproduction did not work, set **Negative result: yes** at the top of your README; the version recorded at your latest submission counts.

Explain not just what you built, but **why it is innovative, who would use it, and whether it could actually work in the real world.**

## Prizes

| Place | Prize |
|---|---|
| **First place team** | $1,000 in Hugging Face credits |
| **Second place team** | $500 in Hugging Face credits |
| **Third place team** | $250 in Hugging Face credits |
| **Top-placing CMU team** | $2,500, shared among the team members. A CMU team is one where more than half of the members are current CMU students (see *Teams*). |

If the top-placing CMU team also places in the top three, it receives that place's Hugging Face credits as well.
