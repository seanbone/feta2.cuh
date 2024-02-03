# A Makefile for development work

builddir ?= build

mkcmd ?= make --directory $(builddir) -j$(j) --no-print-directory


# Default target
all: build demo


$(builddir)/CMakeCache.txt: CMakeLists.txt
	cmake -B $(builddir)           \
	  -DFETA2_BUILD_TESTS=ON       \
	  -DFETA2_BUILD_DEMO=ON        \
	  -DFETA2_DEBUG_MODE=ON        \
	  -DFETA2_BUILD_BENCHMARKS=OFF \
	  -DFETA2_NOT_FETA=ON          \
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

.PHONY: clean
clean:
	rm -rf $(builddir)
