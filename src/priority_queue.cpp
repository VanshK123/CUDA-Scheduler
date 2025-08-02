#include "cuda_scheduler/priority_queue.hpp"
#include <algorithm>
#include <chrono>

namespace cuda_scheduler {

PriorityQueue::PriorityQueue(size_t max_size) 
    : max_size_(max_size)
    , total_enqueued_(0)
    , total_dequeued_(0)
    , total_wait_time_ms_(0.0f) {
    std::fill(std::begin(priority_counts_), std::end(priority_counts_), 0);
}

PriorityQueue::~PriorityQueue() {
    clear();
}

bool PriorityQueue::enqueue(const KernelTask& task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Check if queue is full
    if (queue_.size() >= max_size_) {
        return false;
    }
    
    // Add task to queue
    queue_.push(task);
    
    // Update statistics
    updateStats(task, true);
    
    // Notify waiting threads
    queue_cv_.notify_one();
    
    return true;
}

bool PriorityQueue::dequeue(KernelTask& task) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for task if queue is empty
    if (queue_.empty()) {
        return false;
    }
    
    // Get highest priority task
    task = queue_.top();
    queue_.pop();
    
    // Update statistics
    updateStats(task, false);
    
    return true;
}

size_t PriorityQueue::size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.size();
}

bool PriorityQueue::empty() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return queue_.empty();
}

void PriorityQueue::clear() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Clear the queue
    while (!queue_.empty()) {
        queue_.pop();
    }
    
    // Reset statistics
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    total_enqueued_ = 0;
    total_dequeued_ = 0;
    total_wait_time_ms_ = 0.0f;
    std::fill(std::begin(priority_counts_), std::end(priority_counts_), 0);
}

void PriorityQueue::resize(size_t new_size) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    max_size_ = new_size;
    
    // If new size is smaller, we might need to drop some tasks
    while (queue_.size() > max_size_) {
        queue_.pop();
    }
}

float PriorityQueue::getAverageWaitTime() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (total_dequeued_ == 0) {
        return 0.0f;
    }
    
    return total_wait_time_ms_ / total_dequeued_;
}

PriorityQueue::QueueStats PriorityQueue::getStats() const {
    std::lock_guard<std::mutex> queue_lock(queue_mutex_);
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    
    QueueStats stats;
    stats.total_enqueued = total_enqueued_;
    stats.total_dequeued = total_dequeued_;
    stats.current_size = queue_.size();
    stats.max_size = max_size_;
    stats.avg_wait_time_ms = (total_dequeued_ > 0) ? total_wait_time_ms_ / total_dequeued_ : 0.0f;
    
    std::copy(std::begin(priority_counts_), std::end(priority_counts_), std::begin(stats.priority_counts));
    
    return stats;
}

bool PriorityQueue::waitForTask(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    if (timeout_ms == 0) {
        // Wait indefinitely
        queue_cv_.wait(lock, [this] { return !queue_.empty(); });
        return true;
    } else {
        // Wait with timeout
        return queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !queue_.empty(); });
    }
}

void PriorityQueue::updateStats(const KernelTask& task, bool is_enqueue) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (is_enqueue) {
        total_enqueued_++;
        priority_counts_[static_cast<int>(task.priority)]++;
    } else {
        total_dequeued_++;
        priority_counts_[static_cast<int>(task.priority)]--;
        
        // Calculate wait time
        float wait_time = calculateWaitTime(task);
        total_wait_time_ms_ += wait_time;
    }
}

float PriorityQueue::calculateWaitTime(const KernelTask& task) const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - task.enqueue_time);
    return duration.count() / 1000.0f;  // Convert to milliseconds
}

} // namespace cuda_scheduler 