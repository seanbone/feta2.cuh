# A Makefile for development work

builddir ?= build

j ?= 4

mkcmd ?= make --directory $(builddir) -j$(j) --no-print-directory

cuda_arch_flag = $(if $(cuda_arch),-DCMAKE_CUDA_ARCHITECTURES=$(cuda_arch),)

# Default target
all: build demo


$(builddir)/CMakeCache.txt: CMakeLists.txt
	cmake -B $(builddir)           \
	  -DFETA2_BUILD_TESTS=ON       \
	  -DFETA2_BUILD_DEMO=ON        \
	  -DFETA2_DEBUG_MODE=OFF        \
	  -DFETA2_BUILD_BENCHMARKS=ON  \
	  -DFETA2_NOT_FETA=OFF         \
	  $(cuda_arch_flag)            \
	  .

.PHONY: configure
configure: $(builddir)/CMakeCache.txt

.PHONY: build
build: $(builddir)/CMakeCache.txt
	$(mkcmd) all

$(builddir)/feta2_demo: $(builddir)/CMakeCache.txt
	$(mkcmd) feta2_demo

.PHONY: demo
demo: feta2_demo.cu $(builddir)/feta2_demo
	./$(builddir)/feta2_demo

.PHONY: test
test: $(builddir)/CMakeCache.txt
	$(mkcmd) feta2_tests
	./$(builddir)/tests/feta2_tests

.PHONY: bench
bench: $(builddir)/CMakeCache.txt
	$(mkcmd) feta2_benchmarks
	./$(builddir)/tests/benchmarks/feta2_benchmarks

.PHONY: benchplot
benchplot: $(builddir)/CMakeCache.txt
	$(mkcmd) feta2_benchmarks
	./$(builddir)/tests/benchmarks/feta2_benchmarks \
	 | ./scripts/parseBench.py                      \
	 | ./scripts/plotBench.py

.PHONY: clean
clean:
	rm -rf $(builddir)
