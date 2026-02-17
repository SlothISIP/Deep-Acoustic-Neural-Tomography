# orca-commit

Git commit, push, and sync to GitHub main.

## Trigger Conditions

Activate this skill when the user:
- Says "orca-commit", "/orca-commit", "커밋", "푸시"
- Asks to commit and push current changes
- Says "GitHub 업데이트", "코드 올려"

## Description

Analyzes all current changes, creates a conventional commit, pushes to origin,
and ensures the remote main branch is fully up to date.

## Instructions

When this skill is triggered, execute these steps **exactly in order**:

### Step 1: Assess Current State

Run these commands in parallel:

```bash
git status
git diff --stat
git diff --cached --stat
git log --oneline -3
```

### Step 2: Identify Changes

Categorize all changes into conventional commit types:

| Type | When |
|------|------|
| `feat` | New feature, new file, new capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `docs` | Documentation only |
| `chore` | Build, config, tooling changes |
| `test` | Test additions/modifications |
| `style` | Formatting, no code change |

If changes span multiple types, use the **dominant** type.
Add scope in parentheses: `feat(bem):`, `fix(mesh):`, `docs(phase0):`

### Step 3: Stage Files

Stage all relevant files. **EXCLUDE**:
- `.env`, `credentials.*`, `*.key`, `*.pem` (secrets)
- `__pycache__/`, `*.pyc` (bytecode)
- `*.h5`, `*.hdf5` (BEM data files)
- `*.vtu`, `*.vtk` (large mesh files)
- `docs/figures/*.png`, `docs/figures/*.pdf` (generated figures)

**INCLUDE** everything else: source code, configs, docs, skills, scripts.

Use specific `git add <file>` for each file, NOT `git add -A` or `git add .`.

### Step 4: Create Commit

Draft a commit message following this format:
```
type(scope): concise description

- Bullet point details of key changes
- Reference specific files when helpful

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

Rules:
- Title line: max 72 characters
- Body: wrap at 80 characters
- Use imperative mood: "add", "fix", "update" (not "added", "fixed")
- Focus on **why**, not what

Use HEREDOC for the commit:
```bash
git commit -m "$(cat <<'EOF'
type(scope): description

- detail 1
- detail 2

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### Step 5: Push to Remote

```bash
git push origin main
```

If push fails due to divergence:
1. `git fetch origin`
2. `git rebase origin/main` (NOT merge)
3. Resolve conflicts if any (ask user)
4. `git push origin main`

### Step 6: Verify

```bash
git status
git log --oneline -3
```

Confirm clean working tree and successful push.

### Step 7: Report

Present final summary:

```
## Commit Report

| Item | Value |
|------|-------|
| Commit | `abc1234` |
| Message | `type(scope): description` |
| Files | N files changed |
| Insertions | +NNN |
| Deletions | -NNN |
| Remote | origin/main (up to date) |
```

## Safety Rules

1. **NEVER** force push (`--force`, `-f`)
2. **NEVER** commit secrets (`.env`, credentials, API keys)
3. **NEVER** commit large binaries (HDF5 BEM data, mesh VTK files) unless explicitly asked
4. **NEVER** amend previous commits unless explicitly asked
5. **ALWAYS** show the commit message to user before committing
6. If pre-commit hooks fail: fix issue, re-stage, create NEW commit
