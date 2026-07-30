#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${RAPTOR_ARTIFACT_DIR:-${repo_root}/artifacts/release}"
paper_path="${artifact_root}/2509.11481v2.pdf"
checkpoint_archive="${artifact_root}/raptor-policy-checkpoint.tar.gz"
checkpoint_dir="${artifact_root}/checkpoint"

mkdir -p "${artifact_root}" "${checkpoint_dir}"

curl --fail --location \
  --output "${paper_path}" \
  "https://arxiv.org/pdf/2509.11481v2"

curl --fail --location \
  --output "${checkpoint_archive}" \
  "https://zenodo.org/api/records/17096679/files/raptor-policy-checkpoint.tar.gz/content"

echo "02aff4cc0b90569018e339829c247428  ${checkpoint_archive}" | md5sum --check
tar -xzf "${checkpoint_archive}" -C "${checkpoint_dir}"

sha256sum "${paper_path}" "${checkpoint_archive}" \
  "${checkpoint_dir}/2025-04-19_16-16-17/checkpoint.h5" \
  > "${artifact_root}/SHA256SUMS"

echo "Downloaded release artifacts to ${artifact_root}"
