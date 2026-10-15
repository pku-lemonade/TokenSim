// Discrete-event replay of sampled native operator DAGs on physical resources.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

struct NodeSpec {
    int dep_begin, dep_count, res_begin, res_count, rank, primary;
    double delay, duration, occupancy;
};
struct Completion { long long id; double ready, start, end, released; };
struct Profile {
    std::vector<NodeSpec> nodes;
    std::vector<int> deps, resources;
    std::vector<std::vector<int>> children;
};
struct State { int left; double ready = 0, start = 0; };
struct Job {
    long long id;
    int profile, left;
    double ready, start = -1, released = 0;
    std::vector<State> states;
};
struct Request {
    double ready;
    long long job;
    int node, rank;
    bool operator>(const Request& other) const {
        return std::tie(ready, job, rank, node) >
               std::tie(other.ready, other.job, other.rank, other.node);
    }
    bool same(const Request& other) const {
        return job == other.job && node == other.node;
    }
};
struct Resource {
    bool owned = false;
    std::priority_queue<Request, std::vector<Request>, std::greater<Request>> queue;
    double busy = 0, waited = 0;
    long long operations = 0;
};
struct Event {
    double time;
    int kind;
    long long serial, job;
    int node;
    bool operator>(const Event& other) const {
        return std::tie(time, kind, serial) >
               std::tie(other.time, other.kind, other.serial);
    }
};

class Engine {
public:
    std::string error;
    std::vector<Profile> profiles;
    std::vector<Resource> resources;
    std::unordered_map<long long, Job> jobs;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> events;
    std::set<int> changed;
    double now = 0, begin = NAN, end = INFINITY;
    long long serial = 0, next_job = 0;
    bool arbitration_pending = false;

    void schedule(double time, int kind, long long job = 0, int node = 0) {
        events.push({time, kind, serial++, job, node});
    }
    double overlap(double start, double finish) {
        return std::isnan(begin) ? 0 : std::max(0.0, std::min(end, finish) - std::max(begin, start));
    }
    void wake() {
        if (!arbitration_pending) {
            arbitration_pending = true;
            schedule(now, 3);
        }
    }
    int add_profile(const NodeSpec* nodes, int count, const int* deps,
                    int dep_count, const int* keys, int key_count) {
        if (count <= 0 || dep_count < 0 || key_count < 0)
            throw std::runtime_error("invalid profile sizes");
        Profile profile;
        profile.nodes.assign(nodes, nodes + count);
        if (dep_count) profile.deps.assign(deps, deps + dep_count);
        if (key_count) profile.resources.assign(keys, keys + key_count);
        profile.children.resize(count);
        for (int i = 0; i < count; ++i) {
            const auto& node = nodes[i];
            if (node.dep_begin < 0 || node.dep_count < 0 ||
                node.dep_begin + node.dep_count > dep_count ||
                node.res_begin < 0 || node.res_count < 0 ||
                node.res_begin + node.res_count > key_count ||
                !std::isfinite(node.duration) || !std::isfinite(node.occupancy) ||
                !std::isfinite(node.delay) || node.duration < 0 || node.delay < 0 ||
                node.occupancy < 0 || node.occupancy > node.duration ||
                node.primary < -1 || node.primary >= (int)resources.size())
                throw std::runtime_error("invalid node profile");
            for (int j = 0; j < node.dep_count; ++j) {
                int parent = deps[node.dep_begin + j];
                if (parent < 0 || parent >= count || parent == i)
                    throw std::runtime_error("invalid profile predecessor");
                profile.children[parent].push_back(i);
            }
            for (int j = 0; j < node.res_count; ++j)
                if (keys[node.res_begin + j] < 0 || keys[node.res_begin + j] >= (int)resources.size())
                    throw std::runtime_error("invalid profile resource");
        }
        std::vector<int> incoming(count), ready;
        for (int i = 0; i < count; ++i) {
            incoming[i] = nodes[i].dep_count;
            if (!incoming[i]) ready.push_back(i);
        }
        for (size_t i = 0; i < ready.size(); ++i)
            for (int child : profile.children[ready[i]])
                if (!--incoming[child]) ready.push_back(child);
        if ((int)ready.size() != count) throw std::runtime_error("cyclic profile");
        profiles.push_back(std::move(profile));
        return profiles.size() - 1;
    }
    long long submit(int profile_id, double ready) {
        if (profile_id < 0 || profile_id >= (int)profiles.size() ||
            !std::isfinite(ready) || ready < now)
            throw std::runtime_error("invalid replay submission");
        auto& profile = profiles[profile_id];
        long long id = next_job++;
        Job job{id, profile_id, (int)profile.nodes.size(), ready, -1, 0, {}};
        job.states.resize(profile.nodes.size());
        for (size_t i = 0; i < profile.nodes.size(); ++i) {
            job.states[i].left = profile.nodes[i].dep_count;
            job.states[i].ready = ready;
            if (!job.states[i].left) schedule(ready + profile.nodes[i].delay, 1, id, i);
        }
        jobs.emplace(id, std::move(job));
        return id;
    }
    void grant(const Request& request) {
        auto& job = jobs.at(request.job);
        auto& profile = profiles[job.profile];
        const auto& spec = profile.nodes[request.node];
        auto& state = job.states[request.node];
        state.start = now;
        if (job.start < 0) job.start = now;
        if (spec.primary >= 0)
            resources[spec.primary].waited += overlap(request.ready, now);
        for (int j = 0; j < spec.res_count; ++j) {
            auto& resource = resources[profile.resources[spec.res_begin + j]];
            resource.owned = true;
            resource.queue.pop();
        }
        if (spec.res_count) schedule(now + spec.occupancy, 0, job.id, request.node);
        schedule(now + spec.duration, 2, job.id, request.node);
    }
    void arbitrate() {
        arbitration_pending = false;
        std::vector<Request> candidates;
        for (int key : changed)
            if (!resources[key].queue.empty()) candidates.push_back(resources[key].queue.top());
        changed.clear();
        std::sort(candidates.begin(), candidates.end(), std::greater<Request>());
        std::reverse(candidates.begin(), candidates.end());
        long long previous_job = -1;
        int previous_node = -1;
        for (const auto& request : candidates) {
            if (request.job == previous_job && request.node == previous_node) continue;
            previous_job = request.job; previous_node = request.node;
            const auto& profile = profiles[jobs.at(request.job).profile];
            const auto& spec = profile.nodes[request.node];
            bool available = true;
            for (int j = 0; j < spec.res_count; ++j) {
                const auto& resource = resources[profile.resources[spec.res_begin + j]];
                if (resource.owned || resource.queue.empty() || !resource.queue.top().same(request)) {
                    available = false; break;
                }
            }
            if (available) grant(request);
        }
    }
    int advance(double limit, int external_priority, Completion* result) {
        while (!events.empty()) {
            auto event = events.top();
            if (event.time > limit || (event.time == limit && event.kind >= external_priority)) return 0;
            events.pop(); now = event.time;
            if (event.kind == 3) { arbitrate(); continue; }
            auto& job = jobs.at(event.job);
            auto& profile = profiles[job.profile];
            const auto& spec = profile.nodes[event.node];
            auto& state = job.states[event.node];
            if (event.kind == 1) {
                Request request{now, job.id, event.node, spec.rank};
                if (!spec.res_count) grant(request);
                else {
                    for (int j = 0; j < spec.res_count; ++j) {
                        int key = profile.resources[spec.res_begin + j];
                        resources[key].queue.push(request); changed.insert(key);
                    }
                    wake();
                }
            } else if (event.kind == 0) {
                double busy = overlap(state.start, now);
                for (int j = 0; j < spec.res_count; ++j) {
                    int key = profile.resources[spec.res_begin + j];
                    auto& resource = resources[key];
                    resource.owned = false; resource.busy += busy;
                    if (busy > 0) ++resource.operations;
                    changed.insert(key);
                }
                job.released = std::max(job.released, now); wake();
            } else {
                for (int child : profile.children[event.node]) {
                    auto& next = job.states[child];
                    next.ready = std::max(next.ready, now);
                    if (!--next.left) schedule(next.ready + profile.nodes[child].delay, 1, job.id, child);
                }
                if (!--job.left) {
                    *result = {job.id, job.ready, job.start, now, job.released};
                    jobs.erase(job.id); return 1;
                }
            }
        }
        if (!jobs.empty()) throw std::runtime_error("stalled replay graph");
        return 0;
    }
};

extern "C" {
int ts_node_size() { return sizeof(NodeSpec); }
Engine* ts_create() { return new Engine(); }
void ts_destroy(Engine* engine) { delete engine; }
int ts_resource(Engine* engine) { engine->resources.emplace_back(); return engine->resources.size() - 1; }
void ts_window(Engine* engine, double start, double end) { engine->begin = start; engine->end = end; }
const char* ts_error(Engine* engine) { return engine->error.c_str(); }
int ts_add(Engine* engine, const NodeSpec* nodes, int count, const int* deps, int dep_count, const int* keys, int key_count) {
    try { return engine->add_profile(nodes, count, deps, dep_count, keys, key_count); }
    catch (const std::exception& error) { engine->error = error.what(); return -1; }
}
long long ts_submit(Engine* engine, int profile, double ready) {
    try { return engine->submit(profile, ready); }
    catch (const std::exception& error) { engine->error = error.what(); return -1; }
}
int ts_advance(Engine* engine, double limit, int priority, Completion* result) {
    try { return engine->advance(limit, priority, result); }
    catch (const std::exception& error) { engine->error = error.what(); return -1; }
}
int ts_active(Engine* engine) { return engine->jobs.size(); }
int ts_stats(Engine* engine, int key, double* busy, double* queue, long long* count) {
    if (key < 0 || key >= (int)engine->resources.size()) return -1;
    const auto& resource = engine->resources[key];
    *busy = resource.busy; *queue = resource.waited; *count = resource.operations;
    return 0;
}
}
