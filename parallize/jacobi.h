#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <iostream>

using namespace std;

void jacobi(double** phi, double** aux, double** f, int n, int iternum, MPI_Comm comm)
{
    double h2 = pow(1.0 / ((double)n - 1.0), 2.0);
    int rank;
    MPI_Comm_rank(comm, &rank);
    double* rowbuf = new double[n];
    double* colbuf = new double[n];
    MPI_Status status;
    MPI_Request request_out1, request_in1, request_out3, request_in3;
    MPI_Request request_out2, request_in2, request_out4, request_in4;
    for (int k = 0;k < iternum;k++) {
        //initialize the aux just in case
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                aux[i][j] = 0;
            }
        }
        //we send the information needed first
        if (rank == 0) {//for this subpiece, we recv information
            MPI_Irecv(rowbuf, n, MPI_DOUBLE, 3, k, comm, &request_in1);
            MPI_Irecv(colbuf, n, MPI_DOUBLE, 1, k, comm, &request_in2);
        }
        else if (rank == 2) {//for this subpiece, we recv information
            MPI_Irecv(rowbuf, n, MPI_DOUBLE, 1, k, comm, &request_in3);
            MPI_Irecv(colbuf, n, MPI_DOUBLE, 3, k, comm, &request_in4);
        }
        else if (rank == 3) {
            //for this we send to rank 0 and 2
            //rowbuf = phi[1];
            //initialize the colbuf to send
            for (int i = 0;i < n;i++)
            {
                colbuf[i] = phi[i][n - 2];
                rowbuf[i] = phi[1][i];
            }
            MPI_Isend(rowbuf, n, MPI_DOUBLE, 0, k, comm, &request_out1);
            MPI_Isend(colbuf, n, MPI_DOUBLE, 2, k, comm, &request_out2);
        }
        else {
            //for this we still send
            //rowbuf = phi[n - 2];
            //initialize the colbuf to send
            for (int i = 0;i < n;i++)
            {
                colbuf[i] = phi[i][1];
                rowbuf[i] = phi[n - 2][i];
            }
            MPI_Isend(rowbuf, n, MPI_DOUBLE, 2, k, comm, &request_out3);
            MPI_Isend(colbuf, n, MPI_DOUBLE, 0, k, comm, &request_out4);
        }
        //when data is being transported we calculate the interior point first
        for (int i = 1;i < n - 1;i++) {
            for (int j = 1;j < n - 1;j++) {
                aux[i][j] = (phi[i][j - 1] + phi[i - 1][j] + phi[i + 1][j] + phi[i][j + 1] + h2 * f[i][j]) / 4.0;
            }
        }
        //check if these are done
        if (rank == 0) {
            MPI_Wait(&request_in1, &status);
            MPI_Wait(&request_in2, &status);
            //update up boundary
            for (int j = 1;j < n - 1;j++) {
                aux[n - 1][j] = (phi[n - 1][j - 1] + phi[n - 1][j + 1] + phi[n - 2][j] + rowbuf[j] + h2 * f[n - 1][j]) / 4.0;
            }
            //update right boundary
            for (int i = 1;i < n - 1;i++) {
                aux[i][n - 1] = (phi[i + 1][n - 1] + phi[i - 1][n - 1] + phi[i][n - 2] + colbuf[i] + h2 * f[i][n - 1]) / 4.0;
            }
            //update right up corner
            aux[n - 1][n - 1] = (phi[n - 1][n - 2] + phi[n - 2][n - 1] + colbuf[n - 1] + rowbuf[n - 1] + h2 * f[n - 1][n - 1]) / 4.0;
            //send the boundary value to rank 1 ,3
            MPI_Send(aux[n - 1], n, MPI_DOUBLE, 3, k, comm);
            for (int i = 0;i < n;i++) {
                colbuf[i] = aux[i][n - 1];
            }
            MPI_Send(colbuf, n, MPI_DOUBLE, 1, k, comm);
        }
        else if (rank == 1) {
            MPI_Wait(&request_out3, &status);
            MPI_Wait(&request_out4, &status);
            //recieve boudary value from rank 0,2
            MPI_Recv(colbuf, n, MPI_DOUBLE, 0, k, comm, &status);
            MPI_Recv(rowbuf, n, MPI_DOUBLE, 2, k, comm, &status);
            //update left aux
            for (int i = 0;i < n;i++) {
                aux[i][0] = colbuf[i];
            }
            //update up aux
            for (int j = 0;j < n;j++) {
                aux[n - 1][j] = rowbuf[j];
            }
        }
        else if (rank == 2) {
            MPI_Wait(&request_in3, &status);
            MPI_Wait(&request_in4, &status);
            //update bottom boundary
            for (int j = 1;j < n - 1;j++) {
                aux[0][j] = (phi[0][j - 1] + phi[0][j + 1] + phi[1][j] + rowbuf[j] + h2 * f[0][j]) / 4.0;
            }
            //update left boundary
            for (int i = 1;i < n - 1;i++) {
                aux[i][0] = (phi[i + 1][0] + phi[i - 1][0] + phi[i][1] + colbuf[i] + h2 * f[i][0]) / 4.0;
            }
            //update left bottom corner
            aux[0][0] = (phi[1][0] + phi[0][1] + colbuf[0] + rowbuf[0] + h2 * f[0][0]) / 4.0;
            //send the boundary value to rank 1 ,3
            MPI_Send(aux[0], n, MPI_DOUBLE, 1, k, comm);
            for (int i = 0;i < n;i++) {
                colbuf[i] = aux[i][0];
            }
            MPI_Send(colbuf, n, MPI_DOUBLE, 3, k, comm);
        }
        else if (rank == 3) {
            MPI_Wait(&request_out1, &status);
            MPI_Wait(&request_out2, &status);
            //recieve boudary value from rank 0,2
            MPI_Recv(colbuf, n, MPI_DOUBLE, 2, k, comm, &status);
            MPI_Recv(rowbuf, n, MPI_DOUBLE, 0, k, comm, &status);
            //update right aux
            for (int i = 0;i < n;i++) {
                aux[i][n - 1] = colbuf[i];
            }
            //update bottom aux
            for (int j = 0;j < n;j++) {
                aux[0][j] = rowbuf[j];
            }
        }
        //at last write the new value aux into phi
        for (int i = 0;i < n;i++) {
            for (int j = 0;j < n;j++) {
                phi[i][j] = aux[i][j];
            }
        }


    }
    //free the space we have assigned
    delete[] rowbuf;
    delete[] colbuf;
}

#endif
