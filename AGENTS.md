# tinygrad Architectural Constraints & Lessons Learned
* **Hardware Simulator Evasion via Decorators**: Using @unittest.skipIf based on binary presence silently masks missing dependencies and reports false 100% pass rates. The tests must fail explicitly if the hardware simulator is missing to prevent catastrophic test voids.
* **Massive Skips Masking Bounds**: Massive patterned skips indicate that structural architectural boundaries are likely being masked or evaded rather than organically evaluated.
