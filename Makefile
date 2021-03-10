.PHONY: all clean compile

all: clean generate

clean:
	@echo "Cleaning up..."
	rm -rf build
	rm -rf cmake-build-debug

generate:
	@echo "Generating build..."
	mkdir build
	cd build && cmake .. && make

