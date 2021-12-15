# BaBe: Basecalling Benchmark

Nanopore basecalling guidelines for model benchmarking.

# Todo

- [ ] Add how to calculate phredq scores
    - [ ] CTC greedy
    - [ ] CTC beam
    - [ ] CRF greedy
    - [ ] CRF beam
- [ ] Check normalization methods
    - [ ] mean/sd
    - [ ] pA method
- [ ] Implement metrics with unit tests
    - [ ] Randomness of error
    - [ ] Accuracy-Phredq correlation
    - [ ] Theoretical read difficulty
    - [ ] Experimental read difficulty
    - [ ] Error distribution along the read