NVFLAGS= -O3 -gencode arch=compute_35,code=sm_35 -Xcompiler -fopenmp -rdc=true 

MPI_ROOT=/apps/intel/impi/4.1.0.030

SRCS += \
./main.cu \
./general.cu \
./grands_entiers.cu \
./host_fonctions.cu \
./langford_God_omp16.cu 



# All Target
all: Langford

# Tool invocations
Langford: $(OBJS) 
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(SRCS) -Xcompiler -fopenmp -lcuda -lgomp -O3 -arch=sm_35 -rdc=true -o main
	rm -rf *.d
	rm -rf *.o
	rm -rf *.ii
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	rm -rf Langford
	rm -rf *.d
	rm -rf *.o
	rm -rf *.ii
	rm -rf *~
	rm main

lecture:
	nvcc lecture.cu grands_entiers.cu general.cu -o lecture -rdc=true

.PHONY: all clean dependents

go:
	@echo ' '
	@echo '----- Clean'
	make clean
	@echo ' '
	@echo '----- Build'
	make
	@echo ' '
	@echo '----- Run job'
	sbatch job.sh
	-@echo ' '



secondary-outputs: $(CUBINS)

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
