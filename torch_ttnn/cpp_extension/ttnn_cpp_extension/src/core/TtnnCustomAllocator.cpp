#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"

#include "ttnn_cpp_extension/utils/extension_utils.hpp"
#include <cstring>

at::DataPtr TtnnCustomAllocator::allocate(size_t nbytes) {
    LOGGING("");
    // Do not allocate any cpu space here
    void* data = nullptr;
    return {data, data, &ReportAndDelete, c10::Device(c10::DeviceType::PrivateUse1, 0)};
}

void TtnnCustomAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    // No CPU memory managed; perform a standard memcpy when asked by PyTorch.
    if (count == 0) {
        return;
    }
    TORCH_CHECK(dest != nullptr);
    TORCH_CHECK(src != nullptr);
    std::memcpy(dest, src, count);
}

void TtnnCustomAllocator::ReportAndDelete(void* ptr) {
    LOGGING("");
    TORCH_CHECK(ptr == nullptr)
}

at::DeleterFnPtr TtnnCustomAllocator::raw_deleter() const { return &ReportAndDelete; }

TtnnCustomAllocator& get_ttnn_custom_allocator() {
    static TtnnCustomAllocator allocator;
    return allocator;
}
