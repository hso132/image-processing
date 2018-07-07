NVCC=nvcc
CC=g++
CFLAGS=-std=c++14 -pedantic -Wall
NVCFLAGS=-std=c++14
EXEC=shrinker
all: $(EXEC)
LDFLAGS=-lcuda -lcudart -lpthread -lX11 -Llib/static -ldijkstra

shrinker: shrinker.o tools.o functions.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

tools.o: tools.cpp tools.h
	$(CC) $(CFLAGS) -c -o $@ $<

functions.o: functions.cu functions.h functions.hu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

clean:
	rm -rf *.o
