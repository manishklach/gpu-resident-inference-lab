#ifndef QUEUE_DESC_H
#define QUEUE_DESC_H

#include <cuda_runtime.h>

struct DeviceQueue {
    int* entries;
    int capacity;
    int* head;
    int* tail;
};

__device__ inline bool queue_try_pop(DeviceQueue* queue, int* value) {
    int h = *queue->head;
    int t = *queue->tail;
    if (h == t) return false;
    *value = queue->entries[h];
    *queue->head = (h + 1) % queue->capacity;
    return true;
}

__device__ inline bool queue_try_push(DeviceQueue* queue, int value) {
    int h = *queue->head;
    int t = *queue->tail;
    int next = (t + 1) % queue->capacity;
    if (next == h) return false;
    queue->entries[t] = value;
    *queue->tail = next;
    return true;
}

__host__ inline bool host_queue_try_pop(DeviceQueue* queue, int* value) {
    int h = *queue->head;
    int t = *queue->tail;
    if (h == t) return false;
    *value = queue->entries[h];
    *queue->head = (h + 1) % queue->capacity;
    return true;
}

__host__ inline bool host_queue_try_push(DeviceQueue* queue, int value) {
    int h = *queue->head;
    int t = *queue->tail;
    int next = (t + 1) % queue->capacity;
    if (next == h) return false;
    queue->entries[t] = value;
    *queue->tail = next;
    return true;
}

#endif
