# Check PR Comments

Fetch and address human feedback on a pull request.

## Arguments

$ARGUMENTS should be a PR number (e.g., `152`).

## Instructions

1. Fetch the PR details and all comments:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
gh pr view $ARGUMENTS --json number,title,body,headRefName,files,state,url
gh api repos/{owner}/{repo}/pulls/$ARGUMENTS/comments | jq '.[] | {path: .path, line: .line, body: .body, created_at: .created_at}'
gh pr view $ARGUMENTS --json comments,reviews --jq '{comments: .comments[-10:], reviews: .reviews[-5:]}'
```

2. For each human comment or review:
   - If it's a question → answer in a reply comment
   - If it's a change request → checkout the PR branch, implement the change, re-run QA (`pre-commit run --all-files` and `pytest tests/ -m 'not slow'`), commit, push, and reply confirming
   - If it's a scope change → pause and confirm with the user before proceeding

3. After pushing updates, reply on the PR:

```bash
gh pr comment $ARGUMENTS --body "Addressed feedback: <summary of changes made>."
```

4. Log the action:

```bash
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [Developer] [PR #$ARGUMENTS] [feedback-addressed] <summary>" >> "$REPO_ROOT/ACTIVITY.log"
```
