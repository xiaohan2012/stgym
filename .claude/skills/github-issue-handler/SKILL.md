---
name: github-issue-handler
description: Complete workflow for addressing GitHub issues including fetching issue details, creating branches, investigation, implementation with user approval, quality assurance, and PR creation. Use this skill when users want to work on GitHub issues, fix bugs mentioned in issues, implement features from issues, or when they reference an issue number or GitHub issue URL. Also triggers for requests like 'address this issue', 'work on issue #123', 'fix the bug in issue 456', or 'implement the feature described in this GitHub issue'.
---

# GitHub Issue Handler

A comprehensive skill for systematically addressing GitHub issues from initial investigation through PR creation. This skill implements a complete workflow with quality gates and user approval checkpoints to ensure safe and thorough issue resolution.

## When to Use This Skill

Use this skill whenever:
- User mentions working on a GitHub issue (by number or URL)
- User wants to investigate, reproduce, or fix a bug described in an issue
- User needs to implement a feature request from a GitHub issue
- User asks to "address", "work on", "handle", or "fix" an issue
- User provides GitHub issue URLs or references issue numbers

## Core Workflow

The skill follows an 8-step process with mandatory approval gates:

### 0. Environment Setup

**IMPORTANT: Always activate the virtual environment first:**

```bash
# Activate the STGym virtual environment
pyenv activate stgym
```

This ensures you have access to all project dependencies and tools. If the environment doesn't exist or activation fails, inform the user they need to set up their development environment first.

### 1. Fetch Issue Information

Start by gathering comprehensive issue details using the GitHub CLI.

**Commands to run:**
```bash
# Get detailed issue information in JSON format
gh issue view <issue-number> --json number,title,body,labels,assignees,milestone,state,author,createdAt,updatedAt,url

# Get issue comments if needed for additional context
gh issue view <issue-number> --json comments
```

Parse the returned JSON to understand:
- Issue title and description
- Problem statement or feature requirements
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Labels and priority indicators
- Any existing discussion in comments

### 2. Create Local Branch

Create a descriptive branch name based on the issue:

**Branch naming convention:**
- For bugs: `fix-issue-<number>-<short-description>`
- For features: `feat-issue-<number>-<short-description>`
- For refactoring: `refactor-issue-<number>-<short-description>`

**Commands:**
```bash
# Ensure we're on main and up to date
git checkout main
git pull origin main

# Create and checkout new branch
git checkout -b <branch-name>
```

Example: `git checkout -b fix-issue-123-memory-leak-in-data-loader`

### 3. Investigation Phase

Conduct systematic investigation using Claude's analysis tools:

**For Bug Issues:**
1. **Locate relevant code**: Use Grep to find files related to the issue
2. **Read source files**: Use Read tool to understand current implementation
3. **Reproduce the issue**: Try to recreate the problem locally if possible
4. **Identify root cause**: Analyze code flow and potential failure points

**For Feature Issues:**
1. **Understand requirements**: Parse feature description and acceptance criteria
2. **Explore existing architecture**: Use Glob and Read to understand current codebase structure
3. **Find integration points**: Identify where new code should be added
4. **Check for related existing code**: Look for similar patterns or utilities

**Investigation checklist:**
- [ ] Located relevant source files
- [ ] Understood current behavior/architecture
- [ ] Identified the specific problem or implementation approach
- [ ] Considered edge cases and potential impacts
- [ ] Checked for existing tests that need updating

### 4. Implementation Proposal

**CRITICAL: Get user approval before making any code changes.**

Present a clear implementation plan including:

1. **Problem Summary**: Brief restatement of the issue
2. **Proposed Solution**: High-level approach to fix/implement
3. **Files to Modify**: List of files that will need changes
4. **Testing Strategy**: How the solution will be verified
5. **Potential Risks**: Any concerns or side effects to consider

Wait for explicit user approval with phrases like:
- "Does this approach look correct?"
- "Should I proceed with this implementation?"
- "Please confirm before I make these changes."

**Do not proceed to step 5 without clear user confirmation.**

### 5. Implementation

Execute the approved solution using Claude's editing tools:

- Use Edit tool for targeted changes
- Use MultiEdit for multiple changes in the same file
- Use Write tool only for entirely new files
- Follow existing code conventions and patterns
- Add appropriate comments explaining complex logic
- Update documentation if needed

**Implementation guidelines:**
- Keep changes focused and minimal
- Follow the project's coding style (check existing files for patterns)
- Add error handling where appropriate
- Consider backwards compatibility
- Update type hints and docstrings if the project uses them

### 6. Quality Assurance

Run comprehensive quality checks before committing:

**Ensure virtual environment is active:**
```bash
# Verify environment is activated
pyenv activate stgym
```

**Primary Quality Check (use project's pre-commit setup):**
```bash
pre-commit run --all-files
```

This runs the project's configured linting tools:
- Black (code formatting)
- isort (import sorting)
- autoflake (unused import removal)
- pyupgrade (syntax modernization)
- File hygiene checks
- Security checks

**Testing Validation:**
```bash
# Run all tests
pytest tests/ -v

# Or run specific test modules if they exist
pytest tests/test_<relevant_module>.py -v
```

**Manual Quality Checks:**
- [ ] Code follows project conventions
- [ ] No obvious security issues
- [ ] Error handling is appropriate
- [ ] Changes are well-documented
- [ ] No debug code or temporary changes left in

**If quality checks fail:**
- Fix the issues identified
- Re-run the checks until they pass
- Do not proceed to commit until all checks pass

### 7. Git Operations

Commit and push changes with structured commit messages:

**CRITICAL: Stage only relevant files - NEVER use `git add .`**

```bash
# Stage ONLY the files you intentionally changed
git add .claude/skills/new-skill/SKILL.md
git add path/to/specific/file.py
git add tests/test_new_feature.py

# DO NOT use: git add .
# This stages untracked/temporary files that should not be committed
```

**Commit message format:**
```
<type>(scope): brief description of change

Longer explanation if needed.

Fixes #<issue-number>
```

**Common types:**
- `fix`: Bug fixes
- `feat`: New features
- `refactor`: Code restructuring without behavior change
- `test`: Adding or updating tests
- `docs`: Documentation updates

**Example commands:**
```bash
# Stage specific files only
git add src/data_loader.py tests/test_data_loader.py

# Create structured commit
git commit -m "fix(data-loader): resolve memory leak in batch processing

Fixed issue where DataLoader was not properly releasing memory
between batches, causing gradual memory accumulation during
long training runs.

Fixes #123"

# Push to origin
git push -u origin <branch-name>
```

### 8. Pull Request Creation

Create a comprehensive PR using the GitHub CLI:

```bash
gh pr create --title "<descriptive-title>" --body "$(cat <<'EOF'
## Problem

Brief description of the issue being addressed.

## Solution

Explanation of the implemented solution and approach.

## Changes Made

- List of specific changes
- Files modified
- New functionality added

## Testing

- [ ] Manual testing performed
- [ ] Existing tests pass
- [ ] New tests added (if applicable)

## Review Notes

Any specific areas reviewers should focus on.

Fixes #<issue-number>

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**PR Title Guidelines:**
- Start with the same type as commit message (fix:, feat:, etc.)
- Be descriptive but concise
- Reference the issue number when helpful

**PR Body Requirements:**
- Link to the original issue
- Explain the problem and solution
- List specific changes made
- Include testing information
- Add any special review considerations

## Error Handling and Edge Cases

**If issue fetch fails:**
- Verify issue number/URL is correct
- Check if user has access to the repository
- Ensure GitHub CLI is authenticated (`gh auth status`)

**If branch creation fails:**
- Check if branch already exists
- Ensure git repository is clean
- Verify user has write access

**If quality checks fail:**
- Show specific error messages to user
- Provide guidance on how to fix common issues
- Re-run checks after fixes

**If PR creation fails:**
- Verify branch was pushed successfully
- Check if PR already exists for this branch
- Ensure user has repository access

## Success Criteria

An issue is considered successfully addressed when:

1. ✅ Issue information was correctly fetched and understood
2. ✅ Appropriate branch was created with descriptive name
3. ✅ Thorough investigation was conducted
4. ✅ Implementation plan was approved by user
5. ✅ Code changes were implemented correctly
6. ✅ All quality checks pass (linting, testing)
7. ✅ Changes were committed with proper message
8. ✅ Pull request was created with comprehensive description
9. ✅ Original issue is properly linked and will be closed by the PR

## Reference Files

For additional guidance, consult:
- `references/github_cli.md` - Comprehensive GitHub CLI command reference
- `references/quality_assurance.md` - Detailed quality check procedures and troubleshooting

## Safety Notes

- **Always get user approval** before making code changes
- **Run quality checks** before every commit
- **Never skip testing** if tests exist in the project
- **Ask for clarification** if issue requirements are ambiguous
- **Stop and ask** if you encounter unexpected errors or conflicts
- **CRITICAL: Never use `git add .`** - always stage specific files only
