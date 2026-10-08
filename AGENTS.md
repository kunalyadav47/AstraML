# Antigravity Project Rule: Autonomous Daily 2 PRs

## Core Objective
The primary directive for AstraML is to autonomously deliver **2 Pull Requests every calendar day**. There is **no fixed clock-time condition**—whenever the user opens Antigravity on a given day, the agent must automatically check today's progress and create the required PRs.

---

## Startup Protocol
Whenever Antigravity opens or a new session starts:

1. **Check Today's PR Count**:
   - Query GitHub: `gh pr list --state all --limit 10`
   - Check how many PRs were created on today's calendar date.

2. **If Fewer Than 2 PRs Were Created Today**:
   - **Step A (MANDATORY GATE - Prior PR Must Be Merged)**:
     - Check open PRs: `gh pr list --state open --limit 5`
     - Check latest PR status: `gh pr view --json state,number,title`
     - **IF THE PREVIOUS PR IS STILL OPEN (NOT MERGED)**:
       - **STOP immediately**.
       - Do NOT create a new feature branch or open a new PR.
       - Notify the user that PR #X is awaiting their review/merge, providing the direct PR link.
       - Wait until it is merged into `develop`.
     - **IF AND ONLY IF THE PREVIOUS PR IS CONFIRMED MERGED**:
       - Switch to `develop` and pull latest changes: `git checkout develop && git pull origin develop`
       - Proceed to Step B.
   - **Step B**: Read the next pending feature from `.github/roadmap_backlog.json`.
   - **Step C**: Create the feature branch from `develop`:
     `git checkout -b feature/<feature-name>`
   - **Step D**: Implement the required production files and unit tests according to the architecture plan.
   - **Step E**: Run tests with `pytest` in `.venv` to verify 100% pass rate.
   - **Step F**: Make small, logical conventional commits (`feat(...)`, `test(...)`).
   - **Step G**: Push the branch:
     `git push -u origin feature/<feature-name>`
   - **Step H**: Open the Pull Request on GitHub:
     `gh pr create --base develop --head feature/<feature-name> --title "..." --body "..."`
   - **Step I**: Update `.github/roadmap_backlog.json` and `task.md`.
   - **Repeat** until 2 PRs for today have been opened!

3. **If 2 PRs Have Already Been Created Today**:
   - Inform the user that today's 2-PR milestone is complete.
   - Report the current roadmap status and prepare the next branch for tomorrow.
