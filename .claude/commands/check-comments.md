# Check PR Comments

Fetch and address **unresolved** human feedback on a pull request.

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

Once you have the PR number (`<PR>`) and determined the repo owner/name:

### 1. Fetch unresolved inline review threads (GraphQL)

```bash
gh api graphql -f query='
{
  repository(owner: "<OWNER>", name: "<REPO>") {
    pullRequest(number: <PR>) {
      reviewThreads(first: 50) {
        nodes {
          isResolved
          comments(first: 5) {
            nodes {
              body
              path
              line
              author { login }
              createdAt
            }
          }
        }
      }
    }
  }
}' | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)]'
```

Focus only on **unresolved** threads. Skip threads already marked as resolved.

### 2. Fetch general PR comments

General PR comments (not inline review threads) don't have resolution status. Fetch recent ones:

```bash
gh pr view <PR> --json comments --jq '.comments[-10:]'
```

### 3. Address each unresolved comment

For each unresolved comment or review thread:
- If it's a question → answer in a reply comment
- If it's a change request → checkout the PR branch, implement the change, re-run QA (`pre-commit run --all-files` and `pytest tests/ -m 'not slow'`), commit, push, and reply confirming
- If it's a scope change → pause and confirm with the user before proceeding

### 4. After pushing updates, reply on the PR

```bash
gh pr comment <PR> --body "Addressed feedback: <summary of changes made>."
```

### 5. Log the action

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] [Developer] [PR #<PR>] [feedback-addressed] <summary>" >> "$REPO_ROOT/ACTIVITY.log"
```
