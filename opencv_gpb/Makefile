CC = g++
CFLAGS = `pkg-config --cflags opencv` -I./include/gPb -I./include/sPb -I./include/seg

# LIBS = `pkg-config --libs opencv` -L/opt/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/4.8 -L/usr/lib -larpack -lparpack -L/opt/local/lib/gcc47 -lgfortran
LIBS = `pkg-config --libs opencv` -L/opt/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/4.8 -L/usr/lib/x86_64-linux-gnu -L/usr/lib -larpack -L/opt/local/lib/gcc47 -lgfortran -ltiff
# LIBS = `pkg-config --libs opencv` -L/opt/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/4.8 -L/opt/local/lib/gcc47 -lgfortran

SRC = 	src/main.cpp 		   \
	src/gPb/globalPb.cpp       \
	src/gPb/filters.cpp        \
	src/sPb/buildW.cpp         \
	src/sPb/ic.cpp             \
	src/sPb/affinity.cpp       \
	src/sPb/smatrix.cpp        \
	src/sPb/normCut.cpp        \
	src/seg/watershed.cpp      \
	src/seg/VisWatershed.cpp   \
	src/seg/contour2ucm.cpp    \
	src/seg/ucm_mean_pb.cpp    \
	src/seg/uvt.cpp 

OBJ = gPb

program:
	$(CC) -o $(OBJ) $(SRC) $(CFLAGS) $(LIBS)

clean:
	rm $(OBJ)
