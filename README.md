# emuV - Virtual GPU Memory Emulator

<div align="center">

![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux-green.svg)
![Kernel](https://img.shields.io/badge/kernel-6.x-orange.svg)

**A Linux kernel driver that emulates additional VRAM for NVIDIA GeForce GPUs**

[Features](#features) • [Installation](#installation) • [Configuration](#configuration) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## Overview

emuV is a Linux kernel module that creates a virtual GPU device, allowing you to emulate additional video memory (VRAM) by utilizing system RAM. This can be useful for:

- **Development and testing** - Test applications with different VRAM configurations
- **Memory expansion** - Add virtual VRAM to your existing GPU
- **Learning** - Understand GPU memory management and Linux kernel driver development

## Features

✨ **Multiple GPU Support**
- NVIDIA GeForce RTX 40xx series (4060, 4060 Ti, 4070, 4070 Ti, 4080, 4090)
- NVIDIA GeForce RTX 50xx series (5060, 5070, 5080, 5090)

🔧 **Flexible Configuration**
- Configurable physical and virtual VRAM sizes
- Lazy or eager memory allocation strategies
- Easy configuration via `emuv.conf` file

📊 **Monitoring & Debug**
- Real-time VRAM usage statistics via sysfs
- Debug logging for troubleshooting
- Character device interface for direct access

🚀 **Performance**
- Lazy allocation - memory allocated only when needed
- Multi-threaded support for stress testing
- Efficient memory management

## Supported Systems

- **OS**: Linux (kernel 6.x and above)
- **Architecture**: x86_64
- **GPUs**: NVIDIA GeForce RTX 40xx/50xx series

## Quick Start

```bash
# Clone the repository
git clone https://github.com/bogdanovby/emuV.git
cd emuv

# Edit configuration (optional)
nano emuv.conf

# Build the driver
make

# Load the module
sudo insmod emuv.ko

# Check VRAM info
cat /sys/class/emuv/emuv/vram_info
```

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install build-essential linux-headers-$(uname -r)

# Fedora/RHEL
sudo dnf install kernel-devel gcc make

# Arch Linux
sudo pacman -S linux-headers base-devel
```

### Building

```bash
make clean
make
```

### Loading the Module

```bash
# Load module
sudo insmod emuv.ko

# Verify loading
lsmod | grep emuv
dmesg | tail -20
```

### Installing System-wide

```bash
sudo make install
```

## Configuration

Edit `emuv.conf` to configure the driver:

```ini
# GPU model to emulate
gpu_model=4070

# Physical VRAM size (your actual GPU memory)
physical_vram_gb=8

# Virtual VRAM size (additional memory from system RAM)
virtual_vram_gb=2

# Memory allocation strategy
allocation_strategy=lazy_allocation
```

**Supported GPU models:**
- 40xx series: 4060, 4060Ti, 4070, 4070Ti, 4080, 4090
- 50xx series: 5060, 5070, 5080, 5090

## Usage

### Check VRAM Information

```bash
# Via sysfs
cat /sys/class/emuv/emuv/vram_info

# Via device
sudo cat /dev/emuv
```

### Run Tests

```bash
# Basic test
sudo ./tests/test_emuv.sh

# Stress test (fills all VRAM)
sudo ./tools/run_stress_test.sh

# Python test
sudo python3 ./tests/test_emuv.py
```

### Unload Module

```bash
sudo rmmod emuv
```

## Documentation

- [Installation Guide](docs/INSTALL.ru.md)
- [Stress Testing](docs/STRESS_TEST.md)
- [Configuration Reference](emuv.conf)
- [Development Guide](CONTRIBUTING.md)

## Project Structure

```
emuv/
├── emuv.c              # Main driver source
├── emuv_config.h       # Configuration definitions
├── emuv.conf           # User configuration file
├── Makefile            # Build system
├── README.md           # This file
├── LICENSE             # GPL-2.0 license
├── CONTRIBUTING.md     # Contribution guidelines
├── docs/               # Documentation
│   ├── INSTALL.ru.md
│   └── STRESS_TEST.md
├── tests/              # Test programs
│   ├── test_emuv.c
│   ├── test_emuv.sh
│   ├── test_emuv.py
│   ├── stress_test_emuv.c
│   └── vram_usage_test.c
└── tools/              # Utility scripts
    └── run_stress_test.sh
```

## How It Works

1. **Module Loading**: Driver registers as a character device and creates sysfs interface
2. **Configuration**: Reads settings from `emuv.conf` or uses defaults
3. **Memory Management**: 
   - Physical VRAM: References your actual GPU memory
   - Virtual VRAM: Allocates from system RAM (lazy or eager)
4. **Interface**: Provides `/dev/emuv` and `/sys/class/emuv/` for access

## Performance Considerations

⚠️ **Important Notes:**

- Virtual VRAM uses system RAM, which is slower than actual GPU memory
- Large virtual VRAM allocations may impact system performance
- Recommended to start with 2-4 GB virtual VRAM
- Monitor system memory usage when testing

## Troubleshooting

### Module won't load
```bash
# Check kernel logs
sudo dmesg | grep emuv

# Verify kernel headers
ls /lib/modules/$(uname -r)/build
```

### Device not created
```bash
# Check if module is loaded
lsmod | grep emuv

# Check permissions
ls -l /dev/emuv
```

### Out of memory errors
- Reduce `virtual_vram_gb` in configuration
- Use `lazy_allocation` strategy
- Check available system RAM

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is an educational project. NVIDIA, GeForce, and RTX are trademarks of NVIDIA Corporation. This project is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.

**Use at your own risk.** Loading kernel modules can affect system stability.

## Acknowledgments

- Linux kernel documentation and community
- NVIDIA open-source kernel modules
- All contributors to this project

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/emuv/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/emuv/discussions)

---

<div align="center">

Made with ❤️ by the open-source community

⭐ Star this repo if you find it useful!

</div>
