PATCHES=-I./patches
TCLAP=-I./tclap-1.2.2/include
INCLUDES=-I./include -I./include/partition -I./include/partition/sparse -I./hungarian $(PATCHES) $(TCLAP)
CC=g++ -std=c++17 -O3 -no-pie
FILES=main.cpp graph.cpp utils.cpp sbp.cpp block_merge.cpp common.cpp finetune.cpp
FILES+=partition/partition.cpp partition/partition_triplet.cpp partition/sparse/boost_mapped_matrix.cpp
FILES+=partition/sparse/dict_matrix.cpp partition/sparse/dict_transpose_matrix.cpp evaluate.cpp
FILES+=hung.o

hung.o:
	$(CC) -o hung.o -c hungarian/hungarian.cpp $(PATCHES)

main: hung.o main.cpp
	$(CC) -fopenmp -o main $(FILES) $(INCLUDES) -lstdc++fs

runmain: main
	./main -d ../graph_challenge/data
	rm main hung.o
