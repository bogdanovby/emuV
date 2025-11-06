/*
 * Тестовая программа для проверки работы виртуальной видеокарты
 * Компиляция: gcc -o test_vgpu test_vgpu.c
 * Запуск: sudo ./test_vgpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#define DEVICE_PATH "/dev/emuv"
#define SYSFS_VRAM_INFO "/sys/class/emuv/emuv/vram_info"

void test_device_file(void) {
    printf("=== Тест 1: Проверка символьного устройства ===\n");
    
    int fd = open(DEVICE_PATH, O_RDONLY);
    if (fd < 0) {
        printf("✗ Ошибка открытия %s: %s\n", DEVICE_PATH, strerror(errno));
        return;
    }
    
    printf("✓ Устройство успешно открыто\n");
    
    char buffer[1024];
    ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
    if (bytes_read < 0) {
        printf("✗ Ошибка чтения: %s\n", strerror(errno));
    } else {
        buffer[bytes_read] = '\0';
        printf("✓ Прочитано %zd байт:\n", bytes_read);
        printf("%s\n", buffer);
    }
    
    close(fd);
    printf("\n");
}

void test_sysfs_info(void) {
    printf("=== Тест 2: Проверка sysfs интерфейса ===\n");
    
    FILE *fp = fopen(SYSFS_VRAM_INFO, "r");
    if (!fp) {
        printf("✗ Ошибка открытия %s: %s\n", SYSFS_VRAM_INFO, strerror(errno));
        return;
    }
    
    printf("✓ Файл успешно открыт\n");
    
    char buffer[1024];
    printf("Содержимое файла:\n");
    while (fgets(buffer, sizeof(buffer), fp)) {
        printf("%s", buffer);
    }
    
    fclose(fp);
    printf("\n");
}

void test_device_permissions(void) {
    printf("=== Тест 3: Проверка прав доступа ===\n");
    
    struct stat st;
    if (stat(DEVICE_PATH, &st) < 0) {
        printf("✗ Ошибка получения информации об устройстве: %s\n", strerror(errno));
        return;
    }
    
    printf("✓ Информация об устройстве получена\n");
    printf("  Major: %d\n", major(st.st_rdev));
    printf("  Minor: %d\n", minor(st.st_rdev));
    printf("  Права: %o\n", st.st_mode & 0777);
    printf("  Владелец: UID=%d, GID=%d\n", st.st_uid, st.st_gid);
    printf("\n");
}

void test_sysfs_directory(void) {
    printf("=== Тест 4: Проверка sysfs директории ===\n");
    
    const char *sysfs_dir = "/sys/class/emuv";
    struct stat st;
    
    if (stat(sysfs_dir, &st) < 0) {
        printf("✗ Директория %s не найдена: %s\n", sysfs_dir, strerror(errno));
        return;
    }
    
    if (!S_ISDIR(st.st_mode)) {
        printf("✗ %s не является директорией\n", sysfs_dir);
        return;
    }
    
    printf("✓ Директория найдена\n");
    
    // Попробуем прочитать содержимое
    char command[256];
    snprintf(command, sizeof(command), "ls -la %s/", sysfs_dir);
    printf("Содержимое директории:\n");
    system(command);
    printf("\n");
}

void test_vram_read_write(void) {
    printf("=== Тест 5: Тест чтения/записи VRAM ===\n");
    printf("(Требует реализации ioctl или mmap в драйвере)\n");
    printf("Текущая версия драйвера поддерживает только чтение информации\n");
    printf("\n");
}

int main(void) {
    printf("========================================\n");
    printf("Тест виртуальной видеокарты emuv\n");
    printf("========================================\n\n");
    
    // Проверка прав root
    if (geteuid() != 0) {
        printf("⚠ Предупреждение: Программа запущена не от root\n");
        printf("Некоторые тесты могут не работать\n\n");
    }
    
    test_device_permissions();
    test_device_file();
    test_sysfs_directory();
    test_sysfs_info();
    test_vram_read_write();
    
    printf("========================================\n");
    printf("Тестирование завершено\n");
    printf("========================================\n");
    
    return 0;
}

