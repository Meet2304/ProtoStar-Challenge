# CLAUDE.md

Claude Code reads this file at the start of every session. All project rules live in the single agent
instructions file imported below ([`code/AGENTS.md`](code/AGENTS.md)); they apply to every task,
including exploration in `notebooks/`. Do not create other `CLAUDE.md` or `AGENTS.md` copies.

@code/AGENTS.md

## Claude Code notes

- **Before any web fetch or download of weather data**, state the source and the end date, and cap the
  request at 2018-12-31. If an API or page cannot be limited to dates up to 2018, do not call it. Do not
  look up observed 2019–2025 values (for example, "what was Phoenix's CDD in 2021") for any purpose that
  touches the forecasts.
- **Never write or "correct" a `predicted_value` yourself.** Change the code and regenerate `submission.csv`.
- **Log your work** for the README's *AI Tools and Agents* section: the tool and model, the tasks done,
  and what a person should review. The challenge asks teams to document agent use.
