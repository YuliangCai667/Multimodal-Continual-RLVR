# DEV-2026-09-05-REMOVE-INHERITED-README

## Motivation

The repository root README was inherited largely from the original CPO/MRCL release and presented upstream method claims and usage as this repository's own project overview. It is not suitable as the public-facing description of the CTIR research codebase.

## Changes

- Removed the inherited root `README.md` without replacing it with provisional CTIR claims.
- Kept the license, attribution-bearing source files, experiment documentation, code, and local runtime artifacts unchanged.

## Evaluation

The GitHub repository root should no longer render the inherited CPO/MRCL overview. A dedicated CTIR README can be authored later when the method description and experimental evidence are ready.

## Pitfalls

Until a new project-specific README is added, GitHub will show the repository file listing without a project overview.

## Validation and limits

This is a documentation-only deletion; no runtime tests are required. Publication validation is limited to the staged path list, staged diff whitespace check, remote destination check, and verification that the pushed branch trees do not contain `README.md`.
