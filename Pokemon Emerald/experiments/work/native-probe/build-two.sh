set -eu
set -o pipefail
export PATH=/ucrt64/bin:/usr/bin:$PATH
cd /c/Users/sacch/Documents/Codex/2026-11-24/new-chat/work/upstream-emerald
probe=../native-probe
cflags="-O2 -DNDEBUG -DMODERN=1 -DWASM=1 -DNATIVE=1 -fno-strict-aliasing -fwrapv -Wno-everything -mno-ms-bitfields"
/ucrt64/bin/clang $cflags -I build/wasm -I include/wasm -I include -iquote include -E src/save.c | tools/preproc/preproc.exe -i -g build/assets src/save.c charmap.txt > "$probe/save.i"
python3 tools/native/prepare_source.py "$probe/save.i" "$probe/save.c" coff /ucrt64/bin/clang --target=x86_64-w64-windows-gnu
/ucrt64/bin/clang $cflags -x c -c "$probe/save.c" -o "$probe/save.o"
/ucrt64/bin/clang -O2 -DNDEBUG -c tools/native/native_engine.c -o "$probe/engine.o"
/ucrt64/bin/clang $cflags -I build/wasm -I include/wasm -I include -iquote include -E src/bg.c | tools/preproc/preproc.exe -i -g build/assets src/bg.c charmap.txt > "$probe/bg.i"
python3 tools/native/prepare_source.py "$probe/bg.i" "$probe/bg.c" coff /ucrt64/bin/clang --target=x86_64-w64-windows-gnu
/ucrt64/bin/clang $cflags -x c -c "$probe/bg.c" -o "$probe/bg.o"
find build/native/obj ! -name 'bg.o' -name '*.o' ! -name 'save.o' ! -name 'native_test_game.o' > "$probe/objects.rsp"
find build/native/data -name '*.o' >> "$probe/objects.rsp"
/ucrt64/bin/clang -fuse-ld=lld @"$probe/objects.rsp" "$probe/save.o" "$probe/bg.o" "$probe/engine.o" build/native/bios.o build/native/bench_main.o -lm -o "$probe/fixed-bench.exe"