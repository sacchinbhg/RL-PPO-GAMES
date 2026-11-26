#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include "native_engine.h"
#define STUB(name) void name(void) {}
STUB(NativeInit_contest_ai_scripts) STUB(NativeInit_mystery_event_script_cmd_table)
STUB(NativeInit_multiboot_pokemon_colosseum) STUB(NativeInit_multiboot_ereader) STUB(NativeInit_multiboot_berry_glitch_fix)
STUB(NativeInit_maps) STUB(NativeInit_map_events) STUB(NativeInit_event_scripts) STUB(NativeInit_battle_scripts_1) STUB(NativeInit_battle_scripts_2) STUB(NativeInit_battle_anim_scripts)
STUB(AgbMain) STUB(WasmRunFrame) STUB(WasmRenderFrame)
void *WasmDisplayBuffer(void) {return NULL;} uint32_t WasmCheckSpriteSort(void){return 0;}
uint32_t WasmBattleShortcutCount(void){return 0;} uint32_t WasmBattleShortcutType(uint32_t i){return i;} uint32_t WasmBattleShortcutLabel(uint32_t i){return i;} void WasmBattleShortcutSelect(uint32_t i){}
extern int x,y; extern uint32_t NativePointerToWord(const void*); extern void *NativeDecodePointer(unsigned long);
int main(void){ NativeEngine *e=native_engine_create(); assert(e); assert(x==0 && y==27); x=101; y=202; char stack; void *heap=malloc(4); assert(heap); assert(NativeDecodePointer(NativePointerToWord(&stack))==&stack); assert(NativeDecodePointer(NativePointerToWord(heap))==heap); assert(NativeDecodePointer(NativePointerToWord(&x))==&x); native_engine_destroy(e); e=native_engine_create(); assert(e && x==0 && y==27); native_engine_destroy(e); free(heap);return 0; }
STUB(NativeInit_battle_ai_scripts)
