/*
 * emuV - Virtual GPU Emulation Driver
 * Configuration Header
 */

#ifndef EMUV_CONFIG_H
#define EMUV_CONFIG_H

#include <linux/types.h>

// GPU Model definitions
#define GPU_MODEL_4060    4060
#define GPU_MODEL_4060TI  4061
#define GPU_MODEL_4070    4070
#define GPU_MODEL_4070TI  4071
#define GPU_MODEL_4080    4080
#define GPU_MODEL_4090    4090
#define GPU_MODEL_5060    5060
#define GPU_MODEL_5070    5070
#define GPU_MODEL_5080    5080
#define GPU_MODEL_5090    5090

// GPU Information structure
struct gpu_info {
    int model;
    const char *name;
    u16 device_id;
    u16 vendor_id;
    u64 default_vram_size;  // in bytes
};

// Supported GPU models
static const struct gpu_info supported_gpus[] = {
    // GeForce 40xx series
    {GPU_MODEL_4060,   "NVIDIA GeForce RTX 4060",    0x2882, 0x10DE, 8ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_4060TI, "NVIDIA GeForce RTX 4060 Ti", 0x2803, 0x10DE, 8ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_4070,   "NVIDIA GeForce RTX 4070",    0x2786, 0x10DE, 12ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_4070TI, "NVIDIA GeForce RTX 4070 Ti", 0x2782, 0x10DE, 12ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_4080,   "NVIDIA GeForce RTX 4080",    0x2704, 0x10DE, 16ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_4090,   "NVIDIA GeForce RTX 4090",    0x2684, 0x10DE, 24ULL * 1024 * 1024 * 1024},
    
    // GeForce 50xx series (future models, placeholder IDs)
    {GPU_MODEL_5060,   "NVIDIA GeForce RTX 5060",    0x3000, 0x10DE, 8ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_5070,   "NVIDIA GeForce RTX 5070",    0x3001, 0x10DE, 12ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_5080,   "NVIDIA GeForce RTX 5080",    0x3002, 0x10DE, 16ULL * 1024 * 1024 * 1024},
    {GPU_MODEL_5090,   "NVIDIA GeForce RTX 5090",    0x3003, 0x10DE, 24ULL * 1024 * 1024 * 1024},
};

// Configuration structure
struct emuv_config {
    int gpu_model;
    u64 physical_vram_size;  // in bytes
    u64 virtual_vram_size;   // in bytes
    u64 total_vram_size;     // in bytes
    bool lazy_allocation;
    u16 pci_vendor_id;
    u16 pci_device_id;
    const char *gpu_name;
    int debug_mode;
    int log_level;
};

// Default configuration
#define DEFAULT_GPU_MODEL       GPU_MODEL_4070
#define DEFAULT_PHYSICAL_VRAM   (8ULL * 1024 * 1024 * 1024)  // 8 GB
#define DEFAULT_VIRTUAL_VRAM    (2ULL * 1024 * 1024 * 1024)  // 2 GB
#define DEFAULT_LAZY_ALLOCATION true
#define DEFAULT_DEBUG_MODE      0
#define DEFAULT_LOG_LEVEL       1

// Function prototypes
const struct gpu_info* emuv_get_gpu_info(int model);
int emuv_load_config(struct emuv_config *config, const char *config_file);
void emuv_print_config(const struct emuv_config *config);

#endif /* EMUV_CONFIG_H */

