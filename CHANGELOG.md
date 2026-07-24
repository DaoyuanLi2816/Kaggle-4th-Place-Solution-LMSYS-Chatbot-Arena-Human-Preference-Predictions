# Changelog

## 0.2.0 - 2026-07-23

### Fixed

- Preserve the packer's `max_length` guarantee for custom configurations by
  rejecting negative, non-finite, or empty truncation ratios.
- Reject mismatched batched pair and winner columns instead of silently
  truncating them.
- Validate hard one-hot and soft probability labels before training.
- Return an empty `(0, 3)` array for empty inference inputs and report invalid
  batch sizes clearly.
- Compute evaluation log-loss against the full soft-label distribution.

### Changed

- Read the final assistant message from UltraFeedback-style conversations,
  including system-prompt and multi-turn records.
- Validate training configuration values before loading data or models.
- Create pseudo-label output directories automatically.
- Test Python 3.9, 3.12, and 3.14 in CI and validate release artifacts before
  publishing.
- Update package metadata to current SPDX and Python classifier conventions.
