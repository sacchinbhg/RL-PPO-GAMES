set -eu
export PATH=/ucrt64/bin:/usr/bin:$PATH
cd /c/Users/sacch/Documents/Codex/2026-11-24/new-chat/work/upstream-emerald
/ucrt64/bin/clang -shared -fuse-ld=lld @../native-probe/objects.rsp ../native-probe/save.o build/native/native_engine.o build/native/bios.o build/native/rl_bridge.o -lm -o ../native-probe/fixed.dll