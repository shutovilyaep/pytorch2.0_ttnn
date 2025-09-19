#include "ttnn_cpp_extension/core/TtnnCustomAllocator.hpp"

#include "ttnn_cpp_extension/utils/extension_utils.hpp"
#include <cstring>

c10::DataPtr TtnnCustomAllocator::allocate(size_t nbytes) {
    LOGGING("");
    // Do not allocate any cpu space here
    void* data = nullptr;
    return {data, data, &ReportAndDelete, c10::Device(c10::DeviceType::PrivateUse1, 0)};
}

void TtnnCustomAllocator::ReportAndDelete(void* ptr) {
    LOGGING("");
    TORCH_CHECK(ptr == nullptr)
}

c10::DeleterFnPtr TtnnCustomAllocator::raw_deleter() const { return &ReportAndDelete; }

void TtnnCustomAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
    std::memcpy(dest, src, count);
}

TtnnCustomAllocator& get_ttnn_custom_allocator() {
    static TtnnCustomAllocator allocator;
    return allocator;
}
