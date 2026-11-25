PREFIX     ?= /usr/local
HABANA_LIB ?= /usr/lib/habanalabs
CFLAGS     ?= -O2 -Wall -Wextra
CXXFLAGS   ?= -O2 -Wall -Wextra

all: build/libhpu_mps.so build/scal-xfer build/hpu-mps-top

build:
	mkdir -p $@

build/libhpu_mps.so: src/hpu_mps.cpp src/mps_shm.h | build
	$(CXX) $(CXXFLAGS) -std=c++17 -shared -fPIC -o $@ $< -ldl -lpthread

build/scal-xfer: src/scal-xfer.c | build
	$(CC) $(CFLAGS) -o $@ $< -L$(HABANA_LIB) -lSynapse -Wl,-rpath,$(HABANA_LIB)

build/hpu-mps-top: src/hpu_mps_top.cpp src/mps_shm.h | build
	$(CXX) $(CXXFLAGS) -std=c++17 -o $@ $< -ldl

install: all
	install -Dm755 build/libhpu_mps.so $(DESTDIR)$(PREFIX)/lib/hpu-mps/libhpu_mps.so
	install -Dm755 build/scal-xfer $(DESTDIR)$(PREFIX)/lib/hpu-mps/scal-xfer
	install -Dm755 build/hpu-mps-top $(DESTDIR)$(PREFIX)/lib/hpu-mps/hpu-mps-top
	install -Dm755 hpu-mps $(DESTDIR)$(PREFIX)/bin/hpu-mps

uninstall:
	rm -rf $(DESTDIR)$(PREFIX)/lib/hpu-mps
	rm -f $(DESTDIR)$(PREFIX)/bin/hpu-mps

clean:
	rm -rf build

.PHONY: all install uninstall clean
