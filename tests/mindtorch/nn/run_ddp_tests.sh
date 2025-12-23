#!/bin/bash
# Script to run DDP tests with mpirun

set -e

# Default values
WORLD_SIZE=${1:-2}
TEST_FILE="test_distributed.py"
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running DDP Tests"
echo "=========================================="
echo "World Size: $WORLD_SIZE"
echo "Test File: $TEST_FILE"
echo "Test Directory: $TEST_DIR"
echo "=========================================="

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI or use another multiprocess launcher."
    exit 1
fi

# Change to test directory
cd "$TEST_DIR"

# Run the tests
echo ""
echo "Executing: mpirun -n $WORLD_SIZE python $TEST_FILE"
echo ""

mpirun -n $WORLD_SIZE python "$TEST_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "All tests passed!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Some tests failed (exit code: $EXIT_CODE)"
    echo "=========================================="
fi

exit $EXIT_CODE

