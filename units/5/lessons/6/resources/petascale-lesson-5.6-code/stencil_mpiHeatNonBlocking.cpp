 
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include "mpi.h"


int main (int argc, char *argv[]){
	// Initialize MPI
	MPI::Init(argc,argv);

	// Get the number of processes
	int numP=MPI::COMM_WORLD.Get_size();

	// Get the ID of the process
	int myId=MPI::COMM_WORLD.Get_rank();

	int rows= 4096;
    	int cols= 4096;
    	int numIterations=6500;
    	int speed=1000;
    	//coefficiennst for the stencil
    	const float ctop=0.5;
    	const float cbotton=0.5;
    	const float ceast=0.5;
    	const float cwest=0.5;

	float *data;

	// Only master process initializes the data and broadcasts the data
	// Only master process initializes the data and broadcasts the data
    	if(myId==0){
		data = new float[rows*cols];
		for(int i=0; i<rows; i++)
			for(int j=0; j<cols; j++)
		    		data[i*cols+j] = 0;
	    	//place three initial heat points on the array 
	    	//these heat points are to remain constant
	    	//buff[2]2] temp=10
	    	//buff[3][9] temp =100
	    	//buff[6][6] temp =50
	    	data[2*cols+2] = 10;
	    	data[3*cols+9] = 100;
	    	data[6*cols+6] = 50;
    	}

	// The computation is divided by rows
	int blockRows = rows/numP;
	int myRows = blockRows;

	// For the cases that 'rows' is not multiple of numP
	if(myId < rows%numP){
		myRows++;
	}

	// Measure the current time
	double start = MPI::Wtime();

	// Arrays for the chunk of data to work
	float *myData = new float[myRows*cols];
	float *buff = new float[myRows*cols];

	// The process 0 must specify how many rows are sent to each process
	int *sendCounts;
	int *displs;
	if(!myId){
		sendCounts = new int[numP];
		displs = new int[numP];

		displs[0] = 0;

		for(int i=0; i<numP; i++){

			if(i>0){
				displs[i] = displs[i-1]+sendCounts[i-1];
			}

			if(i < rows%numP){
				sendCounts[i] = (blockRows+1)*cols;
			} else {
				sendCounts[i] = blockRows*cols;
			}
		}
	}

	// Scatter the input matrix
	MPI::COMM_WORLD.Scatterv(data, sendCounts, displs, MPI::FLOAT, myData, myRows*cols, MPI::FLOAT, 0);
	memcpy(buff, myData, myRows*cols*sizeof(float));

	
	// Buffers to receive the rows
	float *prevRow = new float[cols];
	float *nextRow = new float[cols];
	MPI::Status status;
	MPI::Request request[4];
	int iter;
	for(iter=0; iter<numIterations; iter++){
		if(myId > 0){
			// Send the first row to the previous process
			request[0] = MPI::COMM_WORLD.Isend(myData, cols, MPI::FLOAT, myId-1, 0);
			// Receive the previous row from the previous process
			request[1] = MPI::COMM_WORLD.Irecv(prevRow, cols, MPI::FLOAT, myId-1, 0);
		}

		if(myId < numP-1){
			// Send the last row to the next process
			request[2] = MPI::COMM_WORLD.Isend(&myData[(myRows-1)*cols], cols, MPI::FLOAT, myId+1, 0);
			// Receive the next row from the next process
			request[3] = MPI::COMM_WORLD.Irecv(nextRow, cols, MPI::FLOAT, myId+1, 0);
		}

		// Update the main block
		for(int i=1; i<myRows-1; i++){
            		for(int j=1; j<cols-1; j++){
                		// calculate discrete laplacian by averaging 4-neighbourhood
                		buff[i*cols+j]= myData[i*cols+j]+(myData[(i+1)*cols+j]+myData[i*cols+j-1]+myData[i*cols+j+1]+myData[(i-1)*cols+j])/speed;
            		}
        	}

		// Update the first row
		if(myId > 0){
			request[1].Wait(status);
			if(myRows > 1){
				for(int j=1; j<cols-1; j++){
                			buff[j] = myData[j]+(cbotton*myData[cols+j]+cwest*myData[j-1]+ceast*myData[j+1]+ctop*prevRow[j])/speed;		    }

			}
		}
		
		// Update the last row
		if(myId < numP-1){
			request[3].Wait(status);
			if(myRows > 1){
				for(int j=1; j<cols-1; j++){
                			buff[(myRows-1)*cols+j] = myData[j]+(nextRow[j]+myData[(myRows-1)*cols+j-1]+ myData[(myRows-1)*cols+j+1]+myData[(myRows-2)*cols+j])/speed;
            			}
			}
		}

		
		memcpy(myData, buff, myRows*cols*sizeof(float));

	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	MPI::COMM_WORLD.Gatherv(myData, myRows*cols, MPI::FLOAT, data, sendCounts, displs, MPI::FLOAT, 0);

	// Measure the current time
	double end = MPI::Wtime();

	if(!myId){
    		std::cout << "Time with " << numP << " processes: " << end-start << " seconds" << std::endl;
    	
		delete [] data;
	}

	delete [] myData;
	delete [] buff;
	delete [] prevRow;
	delete [] nextRow;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}
