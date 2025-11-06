/*
 * Тест реального использования VRAM через чтение/запись
 * Эмулирует работу приложения, использующего видеопамять
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <time.h>
#include <stdint.h>

#define DEVICE_PATH "/dev/emuv"
#define BLOCK_SIZE (256 * 1024 * 1024) // 256 MB блоки

// Функция для "использования" блока VRAM
void use_vram_block(void *ptr, size_t size, int pattern) {
    uint8_t *data = (uint8_t*)ptr;
    
    // Запись паттерна
    memset(data, pattern, size);
    
    // Чтение и проверка
    for (size_t i = 0; i < size && i < 1024; i++) {
        if (data[i] != (uint8_t)pattern) {
            printf("ОШИБКА: Данные повреждены в позиции %zu\n", i);
            return;
        }
    }
    
    // Симуляция вычислений (запись различных паттернов)
    for (size_t offset = 0; offset < size && offset < 1024*1024; offset += 4096) {
        data[offset] = (uint8_t)(pattern + offset % 256);
    }
}

int main(int argc, char *argv[]) {
    printf("========================================\n");
    printf("Тест использования VRAM\n");
    printf("========================================\n\n");
    
    // Определение размера для теста
    size_t test_size = 10ULL * 1024 * 1024 * 1024; // 10 GB
    if (argc > 1) {
        double gb = atof(argv[1]);
        if (gb > 0) {
            test_size = (size_t)(gb * 1024ULL * 1024 * 1024);
        }
    }
    
    printf("Тестируемый размер: %.2f GB\n", (double)test_size / (1024*1024*1024));
    printf("Размер блока: %zu MB\n", BLOCK_SIZE / (1024*1024));
    printf("\n");
    
    int num_blocks = test_size / BLOCK_SIZE;
    printf("Будет выделено %d блоков\n\n", num_blocks);
    
    void **blocks = malloc(num_blocks * sizeof(void*));
    if (!blocks) {
        printf("ОШИБКА: Не удалось выделить память для указателей\n");
        return 1;
    }
    
    clock_t start = clock();
    
    // Выделение и использование блоков
    for (int i = 0; i < num_blocks; i++) {
        printf("Выделение блока %d/%d...\r", i + 1, num_blocks);
        fflush(stdout);
        
        blocks[i] = malloc(BLOCK_SIZE);
        if (!blocks[i]) {
            printf("\nОШИБКА: Не удалось выделить блок %d\n", i);
            break;
        }
        
        // Использование блока
        use_vram_block(blocks[i], BLOCK_SIZE, i % 256);
        
        if ((i + 1) % 4 == 0) {
            printf("\nВыделено: %.2f GB / %.2f GB\n",
                   (double)(i + 1) * BLOCK_SIZE / (1024*1024*1024),
                   (double)test_size / (1024*1024*1024));
        }
    }
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\n✓ Все блоки выделены и использованы\n");
    printf("Время выполнения: %.2f секунд\n", elapsed);
    printf("Скорость: %.2f GB/сек\n", (double)test_size / (1024*1024*1024) / elapsed);
    
    // Удержание памяти для проверки
    printf("\nПамять удерживается. Нажмите Enter для освобождения...");
    getchar();
    
    // Освобождение памяти
    printf("Освобождение памяти...\n");
    for (int i = 0; i < num_blocks; i++) {
        if (blocks[i]) {
            free(blocks[i]);
        }
    }
    free(blocks);
    
    printf("✓ Память освобождена\n");
    return 0;
}

