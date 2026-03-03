# Dispatch Version Counter Bump Design

## Goal
Move version counter bumping for inplace/mutating ops into the dispatch layer, aligned with Torch semantics. Bump happens after successful execution, and view mutations bump the shared base counter.

## Architecture
- Use schema `mutates` metadata to identify which arguments are mutated.
- Add a dispatch helper that bumps version counters after kernel execution succeeds.
- Handle pipeline and functionalize paths separately to avoid premature or double bumps.

## Data Flow
1. Dispatch resolves schema and executes kernel.
2. If schema indicates mutation, bump the version counter for each mutated argument:
   - For views, bump base’s counter (shared by view/base).
   - Deduplicate targets to avoid double bumps.
3. Functionalize path bumps after writeback, since that is the true mutation point.
4. Pipeline path bumps after the deferred execution finishes.

## Error Handling
- No bump on exceptions (only after successful execution/writeback).
- Mutated non-tensor arguments are ignored.

## Testing
- Add a dispatch-level inplace test that calls `dispatch("add_")` directly and verifies version bump.
- Run existing inplace/version tests for regression.
