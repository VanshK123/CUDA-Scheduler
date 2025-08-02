#pragma once

#include "scheduler.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace cuda_scheduler {

/**
 * @brief Kernel task structure for priority queue
 */
struct KernelTask {
    KernelLaunchParams params;
    Priority priority;
    float predicted_time;
    std::chrono::high_resolution_clock::time_point enqueue_time;
    
    KernelTask() : priority(Priority::NORMAL), predicted_time(0.0f) {}
    KernelTask(const KernelLaunchParams& p, Priority pri, float pred_time) 
        : params(p), priority(pri), predicted_time(pred_time) {
        enqueue_time = std::chrono::high_resolution_clock::now();
    }
    
    // Comparison operator for priority queue (higher priority first)
    bool operator<(const KernelTask& other) const {
        return static_cast<int>(priority) < static_cast<int>(other.priority);
    }
};

/**
 * @brief Priority queue for kernel task management
 * 
 * This class provides:
 * 1. Thread-safe priority queue operations
 * 2. Priority-based task ordering
 * 3. Wait time tracking for performance metrics
 * 4. Queue size management and overflow handling
 */
class PriorityQueue {
public:
    /**
     * @brief Constructor
     * @param max_size Maximum queue size
     */
    explicit PriorityQueue(size_t max_size = 10000);
    
    /**
     * @brief Destructor
     */
    ~PriorityQueue();
    
    /**
     * @brief Enqueue a kernel task
     * @param task Kernel task to enqueue
     * @return true if enqueued successfully
     */
    bool enqueue(const KernelTask& task);
    
    /**
     * @brief Dequeue the highest priority kernel task
     * @param task Output parameter for dequeued task
     * @return true if task was dequeued
     */
    bool dequeue(KernelTask& task);
    
    /**
     * @brief Get current queue size
     * @return Number of tasks in queue
     */
    size_t size() const;
    
    /**
     * @brief Check if queue is empty
     * @return true if queue is empty
     */
    bool empty() const;
    
    /**
     * @brief Clear all tasks from queue
     */
    void clear();
    
    /**
     * @brief Resize the queue
     * @param new_size New maximum size
     */
    void resize(size_t new_size);
    
    /**
     * @brief Get average wait time for tasks
     * @return Average wait time in milliseconds
     */
    float getAverageWaitTime() const;
    
    /**
     * @brief Get queue statistics
     * @return Queue statistics structure
     */
    struct QueueStats {
        size_t total_enqueued;
        size_t total_dequeued;
        size_t current_size;
        size_t max_size;
        float avg_wait_time_ms;
        size_t priority_counts[4];  // Count for each priority level
    };
    
    QueueStats getStats() const;
    
    /**
     * @brief Wait for a task to become available
     * @param timeout_ms Timeout in milliseconds (0 for infinite)
     * @return true if task became available
     */
    bool waitForTask(uint32_t timeout_ms = 0);

private:
    // Priority queue implementation using std::priority_queue
    std::priority_queue<KernelTask> queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Configuration
    size_t max_size_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    size_t total_enqueued_;
    size_t total_dequeued_;
    float total_wait_time_ms_;
    size_t priority_counts_[4];
    
    // Helper methods
    void updateStats(const KernelTask& task, bool is_enqueue);
    float calculateWaitTime(const KernelTask& task) const;
};

} // namespace cuda_scheduler 