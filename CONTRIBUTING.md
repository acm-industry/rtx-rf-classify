# Contributing to this project

## Code Style

### General guidelines
- Try to keep things self-documenting; use good naming of variables. You should only need comments if and only if the underlying task is complex/non-obvious. 
- Keep code patterns (i.e., OOP, functional, multi-paradigm, ...) along with name patterns consistent in a file. This should be noted in the doccumentation (more on this below).
- Use CMakeLists for C++ code.

### Documentation
- Each folder should have an associated `.md` file containing documentation for each function.
- Documentation should be added in the **same PR** as the code it's written for.
- Any public-facing APIs should be clearly documented. You can have internal (hidden) functions or classes, prefer to document these with comments.
    - Functions should have clear in/out arguments and return values documented. The documentation style should be consistent within `.md` files.
    - Classes should have docs describing it's purpose and public-facing APIs. The only important criteria should be that it's purpose and interactivity are obvious.

## Pull Requests
- Each PR should be its own branch. To avoid naming conflicts, name the corresponding branch as `{initials}-{issue number}`.
- A PR should resolve exactly one issue and linked using `closes #{issue number}` syntax in the description of the PR.
- The description should clearly summarize what it achieves and next steps.
- PRs should be squashed into one commit before merging.
- Branches should be deleted after merging into main.