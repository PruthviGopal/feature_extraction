
INC_DIR = -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v6.5\common\inc"
LIB_DIR = -L/cygdrive/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v6.5/lib/Win32
LIB_FILE = -lcuda

EXE_FILE = ../simulation/project.exe
MAIN_EXE_FILE = main.exe

.PHONY: all clean

all: clean $(EXE_FILE)

rebuild: $(MAIN_EXE_FILE)

clean:
	rm -rf ../simulation/*.exe *.o *.out *.obj ../simulation/*.exp ../simulation/*.lib

$(EXE_FILE): $(MAIN_EXE_FILE)
	cp $^ $@

%.exe: %.obj knn.obj remove_point.obj
	nvcc $(LIB_DIR) $(LIB_FILE) $^ -o $@

%.lib: %.obj
	cp $^ $@

%.obj: ../src/%.cu
	nvcc -c $(INC_DIR) $^ -o $@

%.obj: ../src/%.cpp
	nvcc -c $(INC_DIR) $^ -o $@

