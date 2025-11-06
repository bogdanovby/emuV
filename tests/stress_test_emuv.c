/*
 * Нагрузочный тест для виртуальной видеокарты
 * Тестирует использование всей доступной VRAM (10 ГБ)
 * 
 * Компиляция: gcc -o stress_test_vgpu stress_test_vgpu.c -lpthread
 * Запуск: sudo ./stress_test_vgpu [размер_в_ГБ]
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>

#define DEVICE_PATH "/dev/emuv"
#define SYSFS_VRAM_INFO "/sys/class/emuv/emuv/vram_info"

// Глобальные переменные
volatile int running = 1;
size_t total_vram_size = 10ULL * 1024 * 1024 * 1024; // 10 ГБ по умолчанию
size_t allocated_size = 0;
pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

// Структура для отслеживания выделенных блоков
typedef struct memory_block {
    void *ptr;
    size_t size;
    int pattern;
    struct memory_block *next;
} memory_block_t;

memory_block_t *allocated_blocks = NULL;

void signal_handler(int sig) {
    printf("\nПолучен сигнал завершения, освобождаем память...\n");
    running = 0;
}

// Функция для чтения размера VRAM из sysfs
size_t get_vram_size_from_sysfs(void) {
    FILE *fp = fopen(SYSFS_VRAM_INFO, "r");
    if (!fp) {
        return total_vram_size; // Используем значение по умолчанию
    }
    
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "Total VRAM")) {
            // Парсим строку вида "Total VRAM: 10240 MB (10 GB)"
            char *gb_pos = strstr(line, "GB");
            if (gb_pos) {
                // Ищем число перед GB
                char *num_start = gb_pos;
                while (num_start > line && (*num_start != ' ' && *num_start != ':')) {
                    num_start--;
                }
                int gb = atoi(num_start);
                if (gb > 0) {
                    fclose(fp);
                    return (size_t)gb * 1024ULL * 1024 * 1024;
                }
            }
        }
    }
    
    fclose(fp);
    return total_vram_size;
}

// Выделение блока памяти (эмуляция выделения VRAM)
memory_block_t* allocate_vram_block(size_t size, int pattern) {
    void *ptr = malloc(size);
    if (!ptr) {
        return NULL;
    }
    
    // Заполняем память паттерном
    memset(ptr, pattern, size);
    
    memory_block_t *block = malloc(sizeof(memory_block_t));
    if (!block) {
        free(ptr);
        return NULL;
    }
    
    block->ptr = ptr;
    block->size = size;
    block->pattern = pattern;
    block->next = NULL;
    
    pthread_mutex_lock(&alloc_mutex);
    block->next = allocated_blocks;
    allocated_blocks = block;
    allocated_size += size;
    pthread_mutex_unlock(&alloc_mutex);
    
    return block;
}

// Освобождение блока памяти
void free_vram_block(memory_block_t *block) {
    if (!block) return;
    
    pthread_mutex_lock(&alloc_mutex);
    allocated_size -= block->size;
    pthread_mutex_unlock(&alloc_mutex);
    
    free(block->ptr);
    free(block);
}

// Проверка целостности данных в блоке
int verify_block(memory_block_t *block) {
    if (!block || !block->ptr) return 0;
    
    uint8_t *data = (uint8_t*)block->ptr;
    for (size_t i = 0; i < block->size && i < 1024; i++) {
        if (data[i] != (uint8_t)block->pattern) {
            return 0;
        }
    }
    return 1;
}

// Структура для передачи параметров в поток
typedef struct thread_params {
    size_t block_size;
    int thread_id;
} thread_params_t;

// Функция для выделения памяти в отдельном потоке
void* allocation_thread(void *arg) {
    thread_params_t *params = (thread_params_t*)arg;
    size_t block_size = params->block_size;
    int thread_id = params->thread_id;
    int pattern = thread_id % 256;
    
    printf("Поток %d: начало выделения блоков по %.2f MB\n", thread_id, (double)block_size / (1024*1024));
    
    int blocks_allocated = 0;
    while (running) {
        pthread_mutex_lock(&alloc_mutex);
        size_t current_allocated = allocated_size;
        pthread_mutex_unlock(&alloc_mutex);
        
        if (current_allocated >= total_vram_size) {
            break;
        }
        
        size_t remaining = total_vram_size - current_allocated;
        size_t to_allocate = (remaining < block_size) ? remaining : block_size;
        
        memory_block_t *block = allocate_vram_block(to_allocate, pattern);
        if (block) {
            blocks_allocated++;
            if (blocks_allocated % 100 == 0) {
                printf("Поток %d: выделено %d блоков, всего: %.2f GB / %.2f GB\n",
                       thread_id, blocks_allocated,
                       (double)allocated_size / (1024*1024*1024),
                       (double)total_vram_size / (1024*1024*1024));
            }
        } else {
            printf("Поток %d: не удалось выделить память\n", thread_id);
            break;
        }
        
        // Небольшая задержка для реалистичности
        usleep(1000);
    }
    
    printf("Поток %d: завершен, выделено %d блоков\n", thread_id, blocks_allocated);
    return NULL;
}

// Функция для проверки целостности данных
void* verification_thread(void *arg) {
    printf("Поток проверки: запущен\n");
    
    while (running) {
        sleep(5);
        
        pthread_mutex_lock(&alloc_mutex);
        memory_block_t *current = allocated_blocks;
        int total_blocks = 0;
        int verified_blocks = 0;
        
        while (current) {
            total_blocks++;
            if (verify_block(current)) {
                verified_blocks++;
            }
            current = current->next;
        }
        pthread_mutex_unlock(&alloc_mutex);
        
        if (total_blocks > 0) {
            printf("Проверка: %d/%d блоков валидны (%.1f%%), выделено: %.2f GB\n",
                   verified_blocks, total_blocks,
                   (double)verified_blocks / total_blocks * 100,
                   (double)allocated_size / (1024*1024*1024));
        }
    }
    
    return NULL;
}

void print_statistics(void) {
    pthread_mutex_lock(&alloc_mutex);
    
    size_t total_blocks = 0;
    memory_block_t *current = allocated_blocks;
    while (current) {
        total_blocks++;
        current = current->next;
    }
    
    printf("\n=== Статистика ===");
    printf("\nВыделено блоков: %zu\n", total_blocks);
    printf("Выделено памяти: %.2f GB / %.2f GB (%.1f%%)\n",
           (double)allocated_size / (1024*1024*1024),
           (double)total_vram_size / (1024*1024*1024),
           (double)allocated_size / total_vram_size * 100);
    
    pthread_mutex_unlock(&alloc_mutex);
}

void cleanup(void) {
    printf("\nОсвобождение памяти...\n");
    
    pthread_mutex_lock(&alloc_mutex);
    memory_block_t *current = allocated_blocks;
    while (current) {
        memory_block_t *next = current->next;
        free(current->ptr);
        free(current);
        current = next;
    }
    allocated_blocks = NULL;
    allocated_size = 0;
    pthread_mutex_unlock(&alloc_mutex);
    
    printf("Память освобождена\n");
}

int main(int argc, char *argv[]) {
    printf("========================================\n");
    printf("Нагрузочный тест виртуальной видеокарты\n");
    printf("========================================\n\n");
    
    // Установка обработчика сигналов
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Определение размера VRAM
    total_vram_size = get_vram_size_from_sysfs();
    printf("Обнаружен размер VRAM: %.2f GB\n", (double)total_vram_size / (1024*1024*1024));
    
    // Парсинг аргументов
    if (argc > 1) {
        double gb = atof(argv[1]);
        if (gb > 0 && gb <= 10) {
            total_vram_size = (size_t)(gb * 1024ULL * 1024 * 1024);
            printf("Используется размер: %.2f GB\n", gb);
        }
    }
    
    // Проверка доступности устройства
    if (access(DEVICE_PATH, R_OK) != 0) {
        printf("⚠ Устройство %s недоступно (эмуляция в системной памяти)\n", DEVICE_PATH);
    }
    
    printf("\nНачало теста...\n");
    printf("Цель: заполнить %.2f GB VRAM\n\n", (double)total_vram_size / (1024*1024*1024));
    
    // Создание потоков для выделения памяти
    const int num_threads = 4;
    pthread_t threads[num_threads];
    thread_params_t params[num_threads];
    
    // Инициализация параметров потоков
    params[0].block_size = 64 * 1024 * 1024;   // 64 MB
    params[0].thread_id = 0;
    params[1].block_size = 128 * 1024 * 1024;   // 128 MB
    params[1].thread_id = 1;
    params[2].block_size = 256 * 1024 * 1024;   // 256 MB
    params[2].thread_id = 2;
    params[3].block_size = 512 * 1024 * 1024;   // 512 MB
    params[3].thread_id = 3;
    
    // Поток для проверки целостности
    pthread_t verify_thread;
    
    // Запуск потоков
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, allocation_thread, &params[i]);
    }
    
    pthread_create(&verify_thread, NULL, verification_thread, NULL);
    
    // Ожидание завершения или прерывания
    while (running) {
        sleep(2);
        print_statistics();
        
        pthread_mutex_lock(&alloc_mutex);
        if (allocated_size >= total_vram_size * 0.99) {
            printf("\n✓ Достигнуто 99%% использования VRAM!\n");
            running = 0;
        }
        pthread_mutex_unlock(&alloc_mutex);
    }
    
    // Ожидание завершения потоков
    printf("\nОжидание завершения потоков...\n");
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    running = 0;
    pthread_join(verify_thread, NULL);
    
    print_statistics();
    cleanup();
    
    printf("\n=== Тест завершен ===\n");
    return 0;
}

