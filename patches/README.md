# patches/

No patches at the moment: TensorFold 1.2.3 carries what the earlier build of this repository (TensorFold 1.1.5 plus
four patches of ours, commit `3bfc9fc`) added, or matches it in speed. Any `*.patch` file put here is applied, in
name order, to the TensorFold checkout in the Docker build (`git apply`), and `prepare.sh` rebuilds the image when
one changes.
