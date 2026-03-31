# Tinygrad Orchestration Guidelines

## System Rules & Architectural Guidelines

### Build & Orchestration Execution
* **Test Invocation Routine**: Ensure all Python test scripts natively invoke `if __name__ == '__main__': unittest.main()`.

*(Global instructions regarding IPC Muzzling, OOM deadlocks, and Linker limits have been unified to the root AGENTS.md ledger).*
