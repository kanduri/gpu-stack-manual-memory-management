gpu_stack : gpu_stack.cu
	nvcc -std=c++11 -g $^ -o gpu_stack

clean :
	rm -f gpu_stack