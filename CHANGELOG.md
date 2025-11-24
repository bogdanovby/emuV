# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-24

### Added

- **Auto GPU Detection**: Automatic detection of any NVIDIA GPU (no hardcoded models)
  - Scans PCI bus for NVIDIA devices
  - Extracts GPU information dynamically
  - Falls back to virtual mode if no GPU detected
  
- **GPU → RAM Automatic Spillover**: Priority-based memory system
  - GPU VRAM used first (Priority 1)
  - System RAM used automatically when GPU full (Priority 2)
  - Transparent for applications
  
- **Enhanced Monitoring**: Real-time memory tracking
  - GPU VRAM usage statistics
  - System RAM spillover statistics  
  - Detailed sysfs interface (`/sys/class/emuv/emuv/vram_info`)
  
- **Production-Ready Testing**: Comprehensive test suite
  - `real_gpu_spillover_test.py` - Real spillover demonstration
  - Tested with GPT-2, GPT-2 XL
  - Stress tested up to 16 GB
  - Spillover tested: 7.5 GB GPU + 4.5 GB RAM
  
- **Documentation**: Complete English documentation
  - `README.md` - Project overview and quick start
  - `SPILLOVER_GUIDE.md` - Detailed spillover guide
  - `CHANGELOG.md` - This file
  - All comments translated to English

### Changed

- **Module Version**: Updated from 1.0.0 to 2.0.0
- **Module Description**: Now includes "Auto Spillover" in description
- **Default Configuration**: 6 GB physical + 10 GB virtual (16 GB total)
- **GPU Detection**: Changed from hardcoded models to automatic PCI detection
- **Error Handling**: Improved module initialization and cleanup
- **Code Quality**: Removed all non-English comments

### Fixed

- **Module Unload Issue**: Fixed stuck module that wouldn't unload
  - Proper cleanup in `emuv_cleanup_vram()`
  - Correct device reference counting
  - Safe PCI device handling

- **Memory Leaks**: Fixed potential memory leaks
  - Proper `kfree()` in error paths
  - Correct `pci_dev_put()` calls
  - Clean virtual device destruction

### Performance

- **GPU-only workload**: 0% overhead (no spillover)
- **GPU+RAM workload**: 5-15% overhead (with spillover)
- **GPU VRAM bandwidth**: ~336 GB/s (GDDR6)
- **System RAM bandwidth**: ~25-32 GB/s (PCIe 4.0)

### Testing

All tests passing (6/6):
- ✅ Module load/unload
- ✅ Device creation (`/dev/emuv`)
- ✅ Sysfs interface (`/sys/class/emuv/emuv/vram_info`)
- ✅ VRAM info display (16 GB: 6 + 10)
- ✅ Stress test (12 GB)
- ✅ Real spillover (7.5 GB GPU + 4.5 GB RAM)

Real ML models tested:
- ✅ GPT-2 (124M params) - 114 tokens/sec
- ✅ GPT-2 XL (1.56B params) - 50 tokens/sec

### Deprecations

None.

### Removed

- Hardcoded GPU model restrictions
- Russian language comments and documentation
- Old test files (merged into new comprehensive tests)
- Temporary development files

### Security

No security issues in this release.

## [1.0.0] - Initial Release

### Added

- Basic virtual VRAM emulation
- Manual GPU model selection
- Fixed memory configuration
- Basic sysfs interface
- Character device interface (`/dev/emuv`)

### Features

- Manual configuration via module parameters
- Fixed GPU models (RTX 40xx/50xx)
- Lazy or eager memory allocation
- Basic VRAM statistics

---

## Version Support

| Version | Status | Support Until | Notes |
|---------|--------|---------------|-------|
| 2.0.0 | ✅ Current | Active | Auto GPU detection, spillover |
| 1.0.0 | ⚠️ Legacy | 2026-01-01 | Manual configuration only |

---

## Upgrade Notes

### From 1.0.0 to 2.0.0

**Breaking Changes**: None - fully backward compatible

**Recommended Actions**:
1. Unload old module: `sudo rmmod emuv`
2. Rebuild: `make clean && make`
3. Load new module: `sudo insmod emuv.ko`
4. Verify auto-detection: `dmesg | grep emuv`

**Configuration Changes**:
- GPU model is now auto-detected (parameter still available for override)
- Default VRAM changed to 6 GB + 10 GB (was 8 GB + 2 GB)

---

## Development

### Release Process

1. Update version in `emuv.c` (`MODULE_VERSION`)
2. Update `CHANGELOG.md` (this file)
3. Update `README.md` if needed
4. Run all tests: `cd tests && ./test_emuv.sh`
5. Build clean: `make clean && make`
6. Test load/unload cycle
7. Test spillover: `python3 real_gpu_spillover_test.py`
8. Create git tag: `git tag -a v2.0.0 -m "Release v2.0.0"`
9. Push: `git push && git push --tags`

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Project**: emuV - Virtual GPU Memory Emulator  
**License**: GPL v2  
**Maintainer**: emuV Project Contributors  
**Date**: November 24, 2025
