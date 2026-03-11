# Handle GitHub Issue

Address GitHub issues systematically using the complete workflow from investigation through PR creation.

#$ARGUMENTS

Uses the `github-issue-handler` skill located at `.claude/skills/github-issue-handler/` to:

1. Fetch issue details with GitHub CLI
2. Create appropriate branch
3. Investigate and propose solution
4. Implement changes with user approval
5. Run quality checks (pre-commit, tests)
6. Commit and create pull request

Provide an issue number, URL, or description to get started.
