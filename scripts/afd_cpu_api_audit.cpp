// Diagnostic extension of the existing counter; never fit inclusive API time.
#include <cupti.h>
#include <cstdint>
#include <cstring>
#include <time.h>

struct Result { uint64_t wall_ns, api_ns, api_calls, unmatched; };
struct ApiRow { char name[160]; uint64_t ns, calls, copy_kinds; };
static thread_local bool active=false;
static thread_local uint64_t begin_ns, api_begin_ns, api_ns, calls, depth, used, overflow;
static thread_local ApiRow rows[128];
static thread_local const char* outer_name;
static thread_local uint64_t outer_copy_kinds;
static CUpti_SubscriberHandle subscriber;
static uint64_t now_ns() { timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return uint64_t(ts.tv_sec)*1000000000ULL+ts.tv_nsec; }
static void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId,const void* raw) {
    if (!active || (domain!=CUPTI_CB_DOMAIN_RUNTIME_API && domain!=CUPTI_CB_DOMAIN_DRIVER_API)) return;
    auto data=static_cast<const CUpti_CallbackData*>(raw);
    if (data->callbackSite==CUPTI_API_ENTER) {
        if (depth++==0) {
            outer_name=data->functionName; outer_copy_kinds=0;
            if (!strcmp(outer_name,"cudaMemcpyAsync")) {
                auto params=static_cast<const cudaMemcpyAsync_v3020_params*>(data->functionParams);
                outer_copy_kinds=uint64_t(1)<<unsigned(params->kind);
            }
            api_begin_ns=now_ns();
        }
        ++calls;
    } else if (depth) {
        if (--depth==0) {
            auto duration=now_ns()-api_begin_ns; api_ns+=duration;
            unsigned i=0;
            for (;i<used;++i) if (!strcmp(rows[i].name,outer_name)) break;
            if (i==128) { ++overflow; return; }
            if (i==used) { strncpy(rows[i].name,outer_name,159); rows[i].name[159]=0; ++used; }
            rows[i].ns+=duration; ++rows[i].calls; rows[i].copy_kinds|=outer_copy_kinds;
        }
    } else { ++overflow; }
}
extern "C" int afd_init() {
    auto status=cuptiSubscribe(&subscriber,callback,nullptr);
    if(status!=CUPTI_SUCCESS) return int(status);
    status=cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API);
    if(status!=CUPTI_SUCCESS) return int(status);
    return int(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API));
}
extern "C" void afd_begin() {
    api_ns=calls=depth=used=overflow=0; memset(rows,0,sizeof(rows));
    active=true; begin_ns=now_ns();
}
extern "C" void afd_end(Result* result) {
    auto end=now_ns(); active=false; *result={end-begin_ns,api_ns,calls,depth+overflow};
}
extern "C" unsigned afd_api_count() { return used; }
extern "C" const ApiRow* afd_api_row(unsigned index) { return index<used?&rows[index]:nullptr; }
