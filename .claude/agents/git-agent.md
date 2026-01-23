# Git Agent

## Purpose
Handle git operations including push to origin, pull from upstream, and manage commits safely.

## Usage
```
Use the Task tool with subagent_type="general-purpose" and reference the git-agent instructions.

Example prompt:
"Following the git-agent guidelines in .claude/agents/git-agent.md,
push the current changes to origin and then pull latest from ms master."
```

## Git Remotes Configuration

- **origin**: Your fork/development repository (push target)
  - URL: `https://github.com/lvyufeng/mindnlp`
- **ms**: Upstream MindSpore repository (pull source)
  - URL: `https://github.com/mindspore-lab/mindnlp`

Verify remotes:
```bash
git remote -v
```

## Supported Operations

### 1. Push to Origin
```bash
git push origin {branch}
```

### 2. Pull from Upstream (with Rebase)
```bash
git pull --rebase ms master
```

### 3. Create Commits
```bash
git add {files}
git commit -m "message"
```

### 4. Create Feature Branch
```bash
git checkout -b {branch-name}
git push origin {branch-name} -u
```

### 5. Rebase onto Upstream
```bash
git fetch ms
git rebase ms/master
```

### 6. Create Pull Request
```bash
gh pr create --title "Title" --body "Description" --base master --repo mindspore-lab/mindnlp
```

## Safety Rules

### NEVER Do These:
- **Never force push** to shared branches without explicit user permission
- **Never reset commits** on shared branches
- **Never delete branches** without user confirmation
- **Never auto-resolve conflicts** - report them and ask for guidance
- **Never amend commits** that have been pushed
- **Never rebase** branches that others may be working on

### Always Do These:
- **Always pull before pushing** to avoid conflicts
- **Always verify branch** before operations
- **Always report conflicts** to the user
- **Always confirm destructive operations** with user
- **Always check git status** before and after operations

## Workflow Examples

### Example 1: Push Changes and Sync with Upstream
```bash
# Check current status
git status

# Ensure we're on the right branch
git branch

# Push to origin
git push origin master

# Pull latest from upstream
git pull --rebase ms master

# Push rebased changes
git push origin master
```

### Example 2: Create PR from Feature Branch
```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add {files}
git commit -m "Add feature X"

# Push to origin
git push origin feature/my-feature -u

# Create PR
gh pr create --title "Add feature X" --body "Description" --base master --repo mindspore-lab/mindnlp
```

### Example 3: Squash Commits Before PR
```bash
# Fetch latest upstream
git fetch ms

# Rebase interactively (if needed, but prefer non-interactive)
git rebase ms/master

# Or reset and recommit for clean single commit
git reset --soft ms/master
git commit -m "Single commit message"

# Push with force (only to your own branch)
git push origin {branch} --force
```

## Commit Message Format

```
{type}: {brief description}

{detailed description if needed}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Types:
- `fix`: Bug fixes
- `feat`: New features
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

## Conflict Resolution

When conflicts occur:
1. **Report to user** with details of conflicting files
2. **Do not auto-resolve** - conflicts require human judgment
3. **Provide options**:
   - Abort the operation
   - Show the conflicting changes
   - Let user resolve manually

## Error Handling

### Push Rejected
```
! [rejected] master -> master (non-fast-forward)
```
- Pull latest changes first: `git pull --rebase origin master`
- Then retry push

### Merge Conflicts
```
CONFLICT (content): Merge conflict in {file}
```
- Report to user with file list
- Do not auto-resolve

### Authentication Errors
```
fatal: Authentication failed
```
- Check credentials/tokens
- Verify remote URL is correct

## Output Format

After each operation, report:
```
## Git Operation Summary

**Operation**: {what was done}
**Branch**: {current branch}
**Status**: {success/failure}

**Details**:
- {relevant details}

**Next Steps**:
- {suggested next actions if any}
```
