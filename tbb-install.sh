git clone --branch tbb_2020 https://github.com/oneapi-src/oneTBB oneTBB

echo -e "\n\ncompiling..."
(cd oneTBB && make -j)

echo -e "\n\nDone! You can now the output of \"./get-tbb-cflags.sh\" to cflags when compiling"

