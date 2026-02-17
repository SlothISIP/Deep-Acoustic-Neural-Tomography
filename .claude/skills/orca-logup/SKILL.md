# orca-logup

Update CLAUDE.md (summary) and docs/Project_history.md (full detail) with latest changes.

## Trigger Conditions

Activate this skill when the user:
- Says "orca-logup", "/orca-logup", "로그 업데이트", "히스토리 정리"
- Asks to update project documentation with recent changes
- Says "CLAUDE.md 업데이트", "히스토리 추가"

## Description

Two-file documentation update:
1. **CLAUDE.md** - Replace `Implementation Status` section with latest session summary (compact)
2. **docs/Project_history.md** - Append full detailed session log (no summarization)

## Instructions

When this skill is triggered, execute these steps **exactly in order**:

### Step 1: Gather Change Information

Run these commands to understand what changed:

```bash
# Recent commits since last documented session
git log --oneline --since="12 hours ago"

# Full diff from last commit
git diff HEAD~1 --stat

# Current file counts
find . -name "*.py" -not -path "./.claude/*" | wc -l
```

Also read:
- Current `CLAUDE.md` (to find the `Implementation Status` section)
- Current `docs/Project_history.md` (to find the last session number)

### Step 2: Determine Session Number

Read the last session number from `docs/Project_history.md`.
The new session = last session + 1.

If the current session was already partially logged, UPDATE that session instead of creating a new one.

### Step 3: Update CLAUDE.md (Summary Mode)

**Target Section**: `## Implementation Status`

**Rules**:
1. **DELETE** the previous session-specific rows in the Implementation Status table
2. **REPLACE** with current phase states:

```markdown
## Implementation Status

| Phase | Status | Gate Result |
|-------|--------|-------------|
| **0: Foundation Validation** | **COMPLETE** | BEM vs analytical: X.X% error |
| 1: BEM Data Factory | LOCKED | -- |
| ... | ... | ... |

### Session N: [Title] (YYYY-MM-DD)

**Changes**:
- [Bullet point summary of key changes]

### Key Files Modified

| File | Change |
|------|--------|
| `path/to/file` | Brief description |
```

**Constraints**:
- Keep it under 40 lines
- Include quantitative results (%, error values)
- Focus on current state, not history

### Step 4: Update docs/Project_history.md (Full Detail Mode)

**Rules**:
1. **APPEND** a new session section at the end
2. **DO NOT summarize** - include full detail:

```markdown
## Session N: YYYY-MM-DD

### [Session Title]

**Duration**: [estimate]
**Phase**: [current phase]

---

### Changes

[Full detail including:]
- Code changes with file:line references
- BEM solve results with error metrics
- Mesh quality statistics
- Configuration changes
- New files created

---

### Files Created/Modified

| File | Changes |
|------|---------|
| `path/to/file` | Detailed description |

---

### Phase Status

| Phase | Status | Gate Result |
|-------|--------|-------------|
| ... | ... | ... |

---

*Last Updated: YYYY-MM-DD*
```

### Step 5: Update Footer

Update the last line of CLAUDE.md:
```markdown
**Files**: N Python | **Lines**: ~N | **History**: See `docs/Project_history.md`
```

### Step 6: Verify

Read back both files to confirm:
1. CLAUDE.md has clean, current summary
2. Project_history.md has the new session appended
3. Session numbers are sequential

### Step 7: Report

```
## Log Update Report

| File | Action | Lines Changed |
|------|--------|---------------|
| CLAUDE.md | Replaced Implementation Status | +N / -N |
| docs/Project_history.md | Appended Session N | +N lines |

Session N: [Title]
```

## Safety Rules

1. **NEVER** delete previous sessions from Project_history.md (append only)
2. **ALWAYS** replace previous session from CLAUDE.md Implementation Status
3. **NEVER** fabricate metrics - use actual values from logs/output
4. If unsure about a metric value, mark it as "TBD" or "pending"
5. Preserve all content outside the Implementation Status section in CLAUDE.md
