# Quick Start Guide

## Installation

```bash
# Install dependencies
sudo apt-get install build-essential linux-headers-$(uname -r)

# Clone repository
git clone https://github.com/yourusername/emuv.git
cd emuv

# Build the driver
make

# Load the module with default settings (RTX 4070, 8GB physical + 2GB virtual)
sudo insmod emuv.ko

# Or specify custom settings
sudo insmod emuv.ko gpu_model=5090 physical_vram_gb=24 virtual_vram_gb=8
```

## Configuration

Edit `emuv.conf` for persistent settings:

```ini
# GPU model (4060, 4070, 4080, 4090, 5060, 5070, 5080, 5090)
gpu_model=4070

# Memory sizes in GB
physical_vram_gb=8
virtual_vram_gb=2

# Allocation strategy
allocation_strategy=lazy_allocation
```

## Verification

```bash
# Check module is loaded
lsmod | grep emuv

# View VRAM information
cat /sys/class/emuv/emuv/vram_info

# Read through device
sudo cat /dev/emuv

# Check kernel logs
dmesg | grep emuv
```

##Expected Output

```
emuv: Initializing virtual GPU memory emulator
emuv: Emulating NVIDIA GeForce RTX 4070
emuv: VRAM configuration: 8 GB physical + 2 GB virtual = 10 GB total
emuv: Virtual GPU device created successfully
emuv: Device registered as /dev/emuv (major XXX)
emuv: Driver initialized successfully
```

## Testing

```bash
# Run basic tests
sudo ./tests/test_emuv.sh

# Run stress test
sudo ./tools/run_stress_test.sh

# Python test
sudo python3 ./tests/test_emuv.py
```

## Unloading

```bash
sudo rmmod emuv
```

## Supported GPUs

### GeForce 40xx Series
- RTX 4060 (8 GB default)
- RTX 4060 Ti (8 GB default)
- RTX 4070 (12 GB default)
- RTX 4070 Ti (12 GB default)
- RTX 4080 (16 GB default)
- RTX 4090 (24 GB default)

### GeForce 50xx Series
- RTX 5060 (8 GB default)
- RTX 5070 (12 GB default)
- RTX 5080 (16 GB default)
- RTX 5090 (24 GB default)

## Troubleshooting

**Module won't load:**
```bash
sudo dmesg | tail -20
```

**Device not created:**
```bash
ls -l /dev/emuv
ls -l /sys/class/emuv
```

**Out of memory:**
- Reduce `virtual_vram_gb`
- Use `lazy_allocation=1`
- Check available RAM: `free -h`

