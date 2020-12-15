INCLUDES=-I./argparse/include -I./include -I./include/partition -I./include/partition/sparse -I./hungarian
CC=g++ -std=c++17
FILES=main.cpp graph.cpp utils.cpp sbp.cpp block_merge.cpp common.cpp finetune.cpp
FILES+=partition/partition.cpp partition/partition_triplet.cpp partition/sparse/boost_mapped_matrix.cpp
FILES+=partition/sparse/dict_matrix.cpp partition/sparse/dict_transpose_matrix.cpp evaluate.cpp
FILES+=hung.o

.PHONY: init_parser

# argparse is a 3rd-party library for parsing command-line arguments. Added as a submodule so users don't
# have to download it, so it needs to be initialized
init_parser:
	cd argparse
	git submodule init
	git submodule update

hung.o:
	# cd hungarian-algorithm-cpp && make hung.o
	# make hung.o
	# mv hungarian-algorithm-cpp/hung.o ./
	# cd hungarian
	$(CC) -o hung.o -c hungarian/hungarian.cpp
	# mv hung.o ../

main: init_parser hung.o main.cpp
	$(CC) -g -fopenmp -o main $(FILES) $(INCLUDES)

runmain: main
	./main -d ../graph_challenge/data
	rm main hung.o
