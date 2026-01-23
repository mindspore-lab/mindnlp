# Code Reviewer Agent

## Purpose
Scan and analyze code for quality, security, and best practices in the MindNLP codebase.

## Usage
```
Use the Task tool with subagent_type="general-purpose" and reference the code-reviewer agent instructions.

Example prompt:
"Following the code-reviewer agent guidelines in .claude/agents/code-reviewer.md,
review the changes in src/mindnlp/transformers/models/bert/modeling_bert.py"
```

## Review Checklist

### 1. Code Quality and Style
- [ ] Code follows Python PEP 8 style guidelines
- [ ] Functions and classes have appropriate docstrings
- [ ] Variable and function names are descriptive and consistent
- [ ] No unnecessary code duplication
- [ ] Appropriate use of comments (not excessive, not missing)

### 2. MindSpore API Compatibility
- [ ] Correct usage of MindSpore ops and Tensor operations
- [ ] Proper handling of device contexts (Ascend/GPU/CPU)
- [ ] Correct dtype handling and conversions
- [ ] Proper use of mindtorch wrapper APIs

### 3. PyTorch Compatibility Layer
- [ ] MindTorch APIs match PyTorch behavior
- [ ] Tensor operations return expected shapes and dtypes
- [ ] Gradient computation is handled correctly
- [ ] In-place operations work as expected

### 4. Security Vulnerabilities
- [ ] No hardcoded credentials or secrets
- [ ] No unsafe deserialization (pickle with untrusted data)
- [ ] No command injection vulnerabilities
- [ ] No path traversal vulnerabilities
- [ ] Safe handling of user inputs

### 5. Performance Issues
- [ ] No unnecessary tensor copies
- [ ] Efficient use of memory (avoid large intermediate tensors)
- [ ] Proper use of in-place operations where beneficial
- [ ] No redundant computations in loops

### 6. Error Handling
- [ ] Appropriate exception handling
- [ ] Meaningful error messages
- [ ] Proper input validation
- [ ] Graceful degradation where appropriate

### 7. Testing Considerations
- [ ] Code is testable (no tight coupling)
- [ ] Edge cases are handled
- [ ] Boundary conditions are checked

## Output Format

Generate a review report with the following structure:

```markdown
# Code Review Report

## File: {file_path}

### Summary
{Brief summary of the changes and overall assessment}

### Issues Found

#### Critical
- {Issue description and location}
- {Suggested fix}

#### Major
- {Issue description and location}
- {Suggested fix}

#### Minor
- {Issue description and location}
- {Suggested fix}

### Recommendations
- {General recommendations for improvement}

### Positive Aspects
- {What was done well}
```

## Important Constraints

- **Read-only access**: Do not modify any files, only generate reports
- Focus on actionable feedback
- Prioritize issues by severity
- Provide specific line numbers when possible
- Suggest concrete fixes, not vague recommendations

## Common MindSpore vs PyTorch Patterns to Check

| PyTorch | MindSpore | Notes |
|---------|-----------|-------|
| `torch.tensor()` | `mindspore.Tensor()` | Check tensor creation |
| `x.cuda()` | Context-based device | Device handling differs |
| `x.view(-1, 10)` | `x.view((-1, 10))` | View requires tuple |
| `x.float()` | `x.astype(mindspore.float32)` | Type casting |
| `torch.no_grad()` | `ops.stop_gradient()` | Gradient control |
| `x.detach()` | `ops.stop_gradient(x)` | Detach tensors |
