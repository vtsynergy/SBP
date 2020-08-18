INCLUDES=-I./argparse/include -I./include -I./include/partition -I./include/partition/sparse
CC=g++ -std=c++17
FILES=main.cpp graph.cpp utils.cpp sbp.cpp block_merge.cpp common.cpp finetune.cpp
FILES+=partition/partition.cpp partition/partition_triplet.cpp partition/sparse/boost_mapped_matrix.cpp

.PHONY: init_parser

# argparse is a 3rd-party library for parsing command-line arguments. Added as a submodule so users don't
# have to download it, so it needs to be initialized
init_parser:
	cd argparse
	git submodule init
	git submodule update

main: init_parser main.cpp
	$(CC) -g -fopenmp -o main $(FILES) $(INCLUDES)

runmain: main
	./main -d ../graph_challenge/data
	rm main
