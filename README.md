# BRPTD

BRPTD is a robust and privacy-preserving truth discovery implementation for participatory crowdsensing. It provides an end-to-end workflow for data processing, attack simulation, robust truth estimation, residual binning, and proof verification.

## Features

- Processes the IBRL and Beijing Multi-Site Air Quality datasets
- Performs anomaly filtering, reliability estimation, and truth updates with PP-CH
- Implements fixed-point residual computation and residual binning
- Verifies residual bins using Ristretto255 and Bulletproofs
- Exports round-level, trial-level, and summary results

## Project Structure

- `src/brptd/`: data processing, numerical operations, robust aggregation, attack simulation, and proof interfaces
- `native/ristretto_backend/`: Rust proof backend
- `configs/`: runtime configurations
- `tests/`: automated tests

## License

This project is licensed under the GNU Lesser General Public License v3.0.
