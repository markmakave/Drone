CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp
PROG = camera

OPENCV = $(shell pkg-config opencv4 --libs)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(OPENCV) -I ~/dev/Lumina/Autopilot/Depth/include -I ~/dev/Lumina/Shared/include -I /usr/local/include/opencv4