include Makefile
RL_BRIDGE_O := $(NATIVE_BUILD_DIR)/rl_bridge.o
RL_CORE := $(NATIVE_BUILD_DIR)/emerald_rl.$(if $(filter coff,$(NATIVE_FORMAT)),dll,so)

.PHONY: rl-core
rl-core: $(RL_CORE)

$(RL_BRIDGE_O): tools/native/rl_bridge.c $(NATIVE_ENGINE_H)
	@mkdir -p $(dir $@)
	$(NATIVE_CC) $(NATIVE_CFLAGS) -c $< -o $@

$(RL_CORE): $(NATIVE_C_OBJS) $(NATIVE_ENGINE_O) $(NATIVE_BIOS_O) $(NATIVE_DATA_OBJS) $(RL_BRIDGE_O)
	$(NATIVE_CC) $(NATIVE_CFLAGS) $(NATIVE_LDFLAGS) -shared $(if $(filter elf,$(NATIVE_FORMAT)),-Wl$(comma)-Bsymbolic) -o $@ $^ -lm
