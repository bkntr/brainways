# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added
 - Download trained model from huggingface
 - First use setup

## [0.1.9]

### Added
 - Network graph analysis

### Fixed
 - Fix bug where project unable to save after elastix

## [0.1.8.2]

### Fixed
 - Fix bug in cell detection

## [0.1.8]

### Added
 - PLS analysis

## [0.1.7]

### Added
- Add conditions to BrainwaysSubject
- Add BrainwaysProject.calculate_contrast()
- Add BrainwaysProject.possible_contrasts()

### Changed

- Update QuPath version to 0.4.3
- Refactor BrainwaysSubject save file
- Faster and more compact cell counts excel

### Fixed

- Bug where BrainwaysProject.add_subject would not forward atlas and pipeline to subject
- Resuming cell detection now shows correct progress bar percentage

## [0.1.6] - 2023-06-15

### Fixed

- Faster atlas loading time

## [0.1.5] - 2023-06-09

### Added

- Read physical pixel sizes on project creation

## [0.1.4] - 2023-06-08

### Added

- Support changing cell detection parameters

## [0.1.1] - 2023-05-26

### Changed

- Improved brain_mask calculation

## [0.1] - 2023-05-26

### Added

- First version

[unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.1...HEAD
[0.1.8.2]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.8...v0.1.8.2
[0.1.8]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/olivierlacan/keep-a-changelog/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/bkntr/brainways/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/bkntr/brainways/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/bkntr/brainways/compare/v0.1.1...v0.1.4
[0.1.1]: https://github.com/bkntr/brainways/compare/v0.1...v0.1.1
[0.1]: https://github.com/bkntr/brainways/releases/tag/v0.1
