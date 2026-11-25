#include "native_engine.h"
#include <stdint.h>

#ifdef _WIN32
#define RL_API __declspec(dllexport)
#else
#define RL_API __attribute__((visibility("default")))
#endif

extern void RlObserve(int *);
static NativeEngine *engine;

RL_API void rl_close(void)
{
    native_engine_destroy(engine);
    engine = NULL;
}

RL_API int rl_power_on(void)
{
    rl_close();
    engine = native_engine_create();
    if (!engine) return 0;
    native_engine_boot(engine);
    return 1;
}

RL_API int rl_reset(void)
{
    if (!rl_power_on()) return 0;
    native_engine_run_frame(engine);
    native_engine_render(engine);
    return 1;
}

RL_API void rl_frames(uint32_t keys, unsigned int count)
{
    native_engine_set_keys(engine, keys);
    for (unsigned int i = 0; i < count; ++i)
        native_engine_run_frame(engine);
}

RL_API const uint8_t *rl_render(void)
{
    native_engine_render(engine);
    return native_engine_display_buffer(engine);
}

RL_API void rl_observe(int *out) { RlObserve(out); }

static const uint16_t keys[] = {0, 1, 2, 8, 64, 128, 32, 16, 4};
RL_API void rl_step(uint8_t action)
{
    rl_frames(keys[action], 14);
    rl_frames(0, 10);
    native_engine_render(engine);
}

RL_API const uint8_t *rl_screen(void) { return native_engine_display_buffer(engine); }

RL_API int rl_replay(const uint8_t *actions, unsigned int count)
{
    if (!rl_reset()) return 0;
    for (unsigned int i = 0; i < count; ++i) {
        if (actions[i] >= sizeof(keys)/sizeof(keys[0])) return 0;
        rl_step(actions[i]);
    }
    return 1;
}
