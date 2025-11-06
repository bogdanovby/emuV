obj-m += emuv.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f tests/test_emuv tests/stress_test_emuv tests/vram_usage_test

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a
	@echo "Module installed. Load with: sudo modprobe emuv"

uninstall:
	rmmod emuv || true
	rm -f /lib/modules/$(shell uname -r)/extra/emuv.ko
	depmod -a

load:
	@if lsmod | grep -q "^emuv "; then \
		echo "Module already loaded, unloading first..."; \
		rmmod emuv || true; \
	fi
	insmod emuv.ko gpu_model=$(GPU_MODEL) physical_vram_gb=$(PHYS_VRAM) virtual_vram_gb=$(VIRT_VRAM)

unload:
	rmmod emuv || true

reload: unload load

# Parameters for loading (can be overridden)
GPU_MODEL ?= 4070
PHYS_VRAM ?= 8
VIRT_VRAM ?= 2

# Test targets
tests: all
	cd tests && gcc -o test_emuv test_emuv.c
	cd tests && gcc -o stress_test_emuv stress_test_emuv.c -lpthread
	cd tests && gcc -o vram_usage_test vram_usage_test.c
	@echo "Tests compiled"

.PHONY: all clean install uninstall load unload reload tests
