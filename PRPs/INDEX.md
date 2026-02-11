# Product Requirement Prompts (PRPs) Index

## Overview
PRPs are structured templates that guide systematic implementation of complex tasks,
ensuring consistency and completeness through a 4-level validation loop.

## How to Use PRPs

PRPs are **not automatically loaded** by Claude Code. To use a PRP:

1. **Explicit Reference**: Tell Claude to "follow the [PRP-NAME] PRP"
2. **Direct Path**: Provide the full path like "use PRPs/feature/MY-FEATURE.md"
3. **Task Context**: When describing a task, mention which PRP applies

### Example Usage
```
"I need to add a new API endpoint. Please follow the prp_base PRP template."
```

## Templates

### prp_base.md
`PRPs/templates/prp_base.md`
- **Purpose**: Generic implementation template with 4-level validation
- **Use For**: Any new feature, service, or component
- **Sections**: Goal, Context, Blueprint, Validation Loop

### prp_story_task.md
`PRPs/templates/prp_story_task.md`
- **Purpose**: Convert user stories into executable tasks
- **Use For**: Story-driven development from Jira/Linear/etc.
- **Sections**: Story, Tasks, Validation, Completion Checklist

## Project PRPs

<!-- Add project-specific PRPs below as they are created -->

## Quick Reference

**New feature/component:** -> Start with `PRPs/templates/prp_base.md`
**User story implementation:** -> Start with `PRPs/templates/prp_story_task.md`
**Custom PRP:** -> Copy a template and adapt to your needs

## PRP Standards

Every PRP should include:
1. **Context** - Overview and purpose
2. **Implementation Tasks** - Detailed, ordered steps
3. **Validation Loop** - 4 levels (Syntax -> Unit -> Integration -> Domain)
4. **Completion Checklist** - Verification items

---

*Created from template v1.0*
