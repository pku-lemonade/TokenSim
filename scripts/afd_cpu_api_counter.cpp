// Caller-thread CUDA runtime/driver interval union, no device timeline fitting.
#include <cupti.h>
#include <cstdint>
#include <time.h>

struct Result { uint64_t wall_ns, api_ns, api_calls, unmatched; };
static thread_local bool active = false;
static thread_local uint64_t begin_ns, api_begin_ns, api_ns, calls, depth;
static CUpti_SubscriberHandle subscriber;
static uint64_t now_ns() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
static void CUPTIAPI callback(void*, CUpti_CallbackDomain domain,
                              CUpti_CallbackId, const void* raw) {
    if (!active || (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API)) return;
    auto data = static_cast<const CUpti_CallbackData*>(raw);
    if (data->callbackSite == CUPTI_API_ENTER) {
        if (depth++ == 0) api_begin_ns = now_ns();
        ++calls;
    } else if (depth) {
        if (--depth == 0) api_ns += now_ns() - api_begin_ns;
    }
}
extern "C" int afd_init() {
    auto status = cuptiSubscribe(&subscriber, callback, nullptr);
    if (status != CUPTI_SUCCESS) return int(status);
    status = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (status != CUPTI_SUCCESS) return int(status);
    return int(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
}
extern "C" void afd_begin() {
    api_ns = calls = depth = 0;
    active = true;
    begin_ns = now_ns();
}
extern "C" void afd_end(Result* result) {
    const auto end = now_ns();
    active = false;
    *result = {end - begin_ns, api_ns, calls, depth};
}
