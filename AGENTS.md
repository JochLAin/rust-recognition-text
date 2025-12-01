# Repository Guidelines

## Project Structure & Module Organization
The crate root is `Cargo.toml`, with application code inside `src/`. `src/main.rs` currently hosts the CLI entrypoint; add more modules under `src/` and expose binaries via `src/bin/` when needed. Cargo outputs and incremental artifacts live in `target/`—do not edit files there. Keep shared fixtures or sample assets in `assets/` (create it if required) so they remain distinct from compiled output.

## Build, Test, and Development Commands
- `cargo check` — fast validation of compilation errors without generating binaries; run on every edit cycle.
- `cargo fmt` — formats the entire crate using `rustfmt.toml` defaults; ensures consistent diffs.
- `cargo clippy --all-targets --all-features` — static analysis and lint recommendations; treat warnings as actionable.
- `cargo run` — builds and executes the CLI for local experiments; add flags after `--`.
- `cargo test` — runs unit and integration tests; combine with `-- --nocapture` to view stdout when debugging.

## Coding Style & Naming Conventions
Follow idiomatic Rust: four-space indentation, `snake_case` for functions/files, `CamelCase` for types and traits, and `SCREAMING_SNAKE_CASE` for constants. Keep modules focused and favor small functions over monolithic flows. Format with `cargo fmt` before committing, and lint with Clippy to catch style drift or suspicious constructs.

## Testing Guidelines
Unit tests belong near the code they cover inside `#[cfg(test)]` modules, while black-box scenarios should live in `tests/`. Name test functions with the behavior under test (`recognizes_multiline_input`). Aim to keep logical branches covered; add doc tests for public helpers demonstrating usage. Use `cargo test --release` to surface optimizations that affect floating-point or timing behavior.

## Commit & Pull Request Guidelines
Recent history shows short imperative summaries, occasionally prefixed with an emoji tag (`:tada: initialize rust`). Mirror that style: optional emoji, then a concise description under 72 characters. Squash noisy work-in-progress commits locally. Pull requests should describe motivation, list major code paths touched, reference tracking issues, and include `cargo fmt`, `cargo clippy`, and `cargo test` results or screenshots when UI behavior changes.

## Security & Configuration Tips
Never commit secrets or API keys; prefer environment variables loaded at runtime. Treat `target/` as ephemeral and exclude it from patches. If you introduce OCR models or sample datasets, store them outside version control and document download steps so contributors can reproduce your setup without sensitive data leaks.
