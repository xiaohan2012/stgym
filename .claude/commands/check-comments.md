# Check PR Comments

Fetch and address human feedback on a pull request.

## Determine PR Number

If `$ARGUMENTS` is provided, use it as the PR number.

Otherwise, infer the PR number:

1. Check the current branch and find its associated PR:
```bash
gh pr view --json number --jq '.number'
```

2. If that fails (e.g., on `main` or a branch with no PR), check recent conversation context for a PR number.

3. If still unclear, ask the human which PR to check.

## Instructions

Once you have the PR number (`<PR>`):

1. Fetch the PR details and all comments:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
gh pr view <PR> --json number,title,body,headRefName,files,state,url
gh api repos/{owner}/{repo}/pulls/<PR>/comments | jq '.[] | {path: .path, line: .line, body: .body, created_at: .created_at}'
gh pr view <PR> --json comments,reviews --jq '{comments: .comments[-10:], reviews: .reviews[-5:]}'
```

2. For each human comment or review:
   - If it's a question → answer in a reply comment
   - If it's a change request → checkout the PR branch, implement the change, re-run QA (`pre-commit run --all-files` and `pytest tests/ -m 'not slow'`), commit, push, and reply confirming
   - If it's a scope change → pause and confirm with the user before proceeding

3. After pushing updates, reply on the PR:

```bash
gh pr comment <PR> --body "Addressed feedback: <summary of changes made>."
```

4. Log the action:

```bash
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [Developer] [PR #<PR>] [feedback-addressed] <summary>" >> "$REPO_ROOT/ACTIVITY.log"
```
