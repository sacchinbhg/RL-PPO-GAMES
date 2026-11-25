/* This file is compiled as game code, with the port's structure layout. */
#include "global.h"
#include "main.h"
#include "event_data.h"
#include "overworld.h"
#include "pokemon.h"
#include "load_save.h"
#include "constants/flags.h"
#include "constants/vars.h"
#include "rl_map_counts.h"

extern int RlTitlePhase(void);
extern int RlMenuPhase(void);
extern int RlNamingPhase(void);
extern void BattleMainCB2(void);

/* Nine values match the policy input. The four extra values are for audits. */
void RlObserve(int *out)
{
    int i, group, number, intro, phase = RlTitlePhase();
    const struct MapHeader *header;
    for (i = 0; i < 13; ++i) out[i] = 0;
    out[1] = out[2] = -1;
    if (!phase) phase = RlMenuPhase();
    if (!phase) phase = RlNamingPhase();
    if (gMain.callback2 == CB2_Overworld) phase = 5;
    if (gMain.callback2 == BattleMainCB2) phase = 6;
    out[0] = phase;
    if (gSaveBlock1Ptr == NULL || gSaveBlock2Ptr == NULL) return;
    intro = VarGet(VAR_LITTLEROOT_INTRO_STATE);
    if (intro < 1 || intro > 7) return;
    group = gSaveBlock1Ptr->location.mapGroup;
    number = gSaveBlock1Ptr->location.mapNum;
    if (group < 0 || group >= (int)ARRAY_COUNT(rlMapCounts)
        || number < 0 || number >= rlMapCounts[group]) return;
    header = Overworld_GetMapHeaderByGroupAndId(group, number);
    if (!header || !gMapHeader.mapLayoutId || gMapHeader.mapLayoutId != header->mapLayoutId
        || gMapHeader.events != header->events) return;
    out[0] = phase ? phase : 7;
    out[1] = group;
    out[2] = number;
    out[3] = gSaveBlock1Ptr->pos.x >> 1;
    out[4] = gSaveBlock1Ptr->pos.y >> 1;
    out[5] = intro > 1 ? intro - 1 : 1;
    out[6] = FlagGet(FLAG_SYS_CLOCK_SET) != 0;
    out[7] = gPlayerPartyCount >= 1 && gPlayerPartyCount <= 6;
    for (i = 0; i < 8; ++i)
        if (FlagGet(FLAG_BADGE01_GET + i)) out[8] |= 1 << i;
    if (FlagGet(FLAG_SYS_GAME_CLEAR)) out[8] |= 256;
    out[9] = VarGet(VAR_LITTLEROOT_TOWN_STATE) >= 1 && VarGet(VAR_LITTLEROOT_RIVAL_STATE) >= 3;
    out[10] = VarGet(VAR_ROUTE101_STATE);
    out[11] = gPlayerPartyCount;
    out[12] = gSaveBlock2Ptr->playerGender;
}
