# Contributing to emuV

Thank you for your interest in contributing to emuV! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/emuv/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (kernel version, GPU model, etc.)
   - Relevant logs (`dmesg` output)

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create an issue describing:
   - The feature or enhancement
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a branch** for your feature or fix
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow the coding style (see below)
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   make clean
   make
   sudo insmod emuv.ko
   # Run tests
   sudo ./tests/test_emuv.sh
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Create a Pull Request**
   - Describe your changes
   - Reference any related issues
   - Include test results

## Coding Style

### C Code

Follow Linux kernel coding style:
- Tabs for indentation (8 spaces wide)
- 80-character line limit where reasonable
- K&R style braces
- Descriptive variable names
- Comments for complex logic

```c
/* Good example */
static int emuv_init_device(struct emuv_device *dev)
{
    int ret = 0;
    
    if (!dev) {
        pr_err("emuV: Invalid device pointer\n");
        return -EINVAL;
    }
    
    /* Initialize device structure */
    ret = allocate_resources(dev);
    if (ret) {
        pr_err("emuV: Failed to allocate resources\n");
        return ret;
    }
    
    return 0;
}
```

### Naming Conventions

- Functions: `emuv_` prefix, lowercase with underscores
- Structures: `emuv_` prefix, lowercase with underscores
- Macros/Constants: UPPERCASE with underscores
- Static functions: `static` keyword, no need for prefix

### Logging

Use appropriate log levels:
```c
pr_err("emuV: Critical error\n");        // Errors
pr_warn("emuV: Warning message\n");      // Warnings
pr_info("emuV: Informational\n");        // Important info
pr_debug("emuV: Debug details\n");       // Debug only
```

### Error Handling

Always check return values and handle errors:
```c
ret = some_function();
if (ret) {
    pr_err("emuV: Operation failed: %d\n", ret);
    goto error_cleanup;
}
```

## Testing

### Before Submitting

1. **Build test**
   ```bash
   make clean
   make
   ```

2. **Load test**
   ```bash
   sudo insmod emuv.ko
   sudo rmmod emuv
   ```

3. **Functional tests**
   ```bash
   sudo ./tests/test_emuv.sh
   ```

4. **Check logs**
   ```bash
   dmesg | grep emuV
   ```

### Adding Tests

If adding new features, please include tests in the `tests/` directory.

## Documentation

- Update README.md for user-facing changes
- Add/update comments in code for complex logic
- Update configuration examples if config changes
- Add documentation in `docs/` for major features

## GPU Support

### Adding New GPU Models

To add support for a new GPU model:

1. **Update `emuv_config.h`**:
   ```c
   #define GPU_MODEL_XXXX    XXXX
   
   // In supported_gpus array:
   {GPU_MODEL_XXXX, "NVIDIA GeForce RTX XXXX", 0xDEVICE_ID, 0x10DE, DEFAULT_VRAM},
   ```

2. **Test the new model**:
   ```bash
   # Edit emuv.conf
   gpu_model=XXXX
   
   # Load and test
   sudo insmod emuv.ko
   cat /sys/class/emuv/emuv/vram_info
   ```

3. **Update documentation** in README.md

## Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
- Reference issues: "Fixes #123" or "Closes #456"
```

Examples:
```
Add support for RTX 5090

Implement RTX 5090 GPU emulation with proper device ID
and memory configuration.

Closes #42
```

## Review Process

1. Automated checks will run on your PR
2. Maintainers will review your code
3. Address any requested changes
4. Once approved, your PR will be merged

## Questions?

- Open an issue for technical questions
- Use Discussions for general questions
- Check existing documentation first

## License

By contributing, you agree that your contributions will be licensed under the GPL-2.0 License.

---

Thank you for contributing to emuV! 🎉

