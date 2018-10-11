#include <stdlib.h> 
#include <assert.h> 
#include <float.h> 
#include <math.h>
#include <stdio.h>
#include <time.h>

#define THREAD_DIM 256

#define CHECK(call) { \
		const cudaError_t error = call; \
		if (error != cudaSuccess) { \
		printf("Error: %s:%d, ", __FILE__, __LINE__); \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

void plot(double *data_points, int n, int m, int *labels, int k);
void countClusters(int *count, int k, int *labels, int n);
void init_centroids(double *data, int d, int k, double *centroids);

// O(n)+O(k) = O(n)
void countClusters(int *count, int k, int *labels, int n){
	int i;
	for(i = 0; i < k; i++){
		count[i]=0;
	}

	for(int j = 0; j < n; j++){
		count[labels[j]]+=1;
	}
};

__device__ double euclidean_distance(int d, double *point1, double *point2){
	double distance = 0;
	int j;
	for(j = 0; j < d; j++){
		distance += sqrt(powf(point1[j] - point2[j], 2));
	}
	return distance;
}

__device__ double atomicAddD(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Complessità: O(kd)
__global__ void finding_closest(double *data, int n, int d, double *centroids, int k, int *labels, double *min_distances, double *tmp_centroids, int *counts ){
	
	int dim = d;
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int j = 0;
	double min_distance = DBL_MAX;
	double newDistance = 0;
	double *idata;
	int best_cluster = 0;
	double *c;

	if( thread_index < n ){

		idata = &data[thread_index * dim];

		for(j = 0; j < k; j++){
			//calcolo distanza tra data e clusters
			newDistance = euclidean_distance(dim, idata, &centroids[j * dim]);
		
			if(newDistance < min_distance){
				min_distance = newDistance;
				best_cluster = j;
			}
		}
		min_distances[thread_index] = min_distance;
		labels[thread_index] = best_cluster;

		c = &tmp_centroids[best_cluster * dim];
		for(int i=0;i<dim;i++){
			atomicAddD(&c[i], idata[i]);
		}
  		atomicAdd(&counts[best_cluster], 1);
	}
}

void init_centroids(double *data, int d, int k, double *centroids){
	double *ci;
	double *di;
	int i,j,h;
	for(i = 0, h= 5; i < k; i++, h += 5*i){
		ci = &centroids[i * d];
		di = &data[i * d];
		for (j = 0; j < d; j++){
			ci[j] = di[j];
		}
	}
}

int main(int argc, char *argv[]) {

	#define NUM_POINTS  5000
	#define THRESHOLD 1e-40
	#define DATASET_NAME "data/dataset.txt"
	#define DIM 2

	int i = 0, j = 0, k;
	double a = 0, b = 0;
	
    FILE *file;
    
    double *host_tmp_centroids;
    double *dev_tmp_centroids;
    double *host_min_distances;
  	
	double *host_centroids;
	double *host_data_points;
    int *host_labels;
    int *host_counts;

	double *dev_min_distances;
	double *dev_data_points;
	double *dev_centroids;
	int *dev_labels;
    int *dev_counts;

	// Contiene il numero di punti appartenenti al i-esimo cluster
	//int *count;


	if( argc == 2 ) {
      printf("Numero di cluster %s\n", argv[1]);
      printf("Numero di punti   %d\n", NUM_POINTS);
   	}else{
   		return -1;
   	}

   	k = atoi(argv[1]);

   	if( k > NUM_POINTS){
   		printf("ERRORE: il numero di cluster è superiore al numero di punti\n");
   		return -1;
   	}

   	//Allocazione della memoria HOST
	//count = (int*)calloc( k, sizeof(int) );
	host_labels = (int*)calloc(NUM_POINTS, sizeof(int));
	host_data_points = (double*)malloc(NUM_POINTS*DIM*sizeof(double));
	host_centroids = (double*)malloc(k*DIM*sizeof(double));
	host_min_distances = (double*)calloc(NUM_POINTS, sizeof(double));
	host_tmp_centroids = (double*)malloc(k*DIM*sizeof(double));
	host_counts = (int*)malloc(k*sizeof(int));

	// Allocazione della memoria DEVICE
	cudaMalloc( (void**)&dev_min_distances, NUM_POINTS*sizeof(double) );
	cudaMalloc( (void**)&dev_data_points, NUM_POINTS*DIM*sizeof(double) );
	cudaMalloc( (void**)&dev_centroids, k*DIM*sizeof(double) );
	cudaMalloc( (int**)&dev_labels, NUM_POINTS*sizeof(int) );
	cudaMalloc( (int**)&dev_tmp_centroids, k*DIM*sizeof(double) );
	cudaMalloc( (int**)&dev_counts, k*sizeof(int) );
	
	// Apre il dataset salvato nel file specificato e lo carica in memoria
	file = fopen(DATASET_NAME,"r");
	i=0;
  	while (fscanf(file, "%lf %lf", &a, &b) != EOF && i < NUM_POINTS*DIM) {
  		host_data_points[i] = a;
  		host_data_points[i+1] = b;
  		i += DIM;
	}

	// Inizializzazione dei centroidi
	init_centroids(host_data_points, DIM, k, host_centroids);

	// Copia dei dati dalla memoria HOST alla memoria DEVICE
	CHECK(cudaMemcpy(dev_data_points, host_data_points, NUM_POINTS*DIM*sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_labels, host_labels, NUM_POINTS * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_min_distances, host_min_distances, NUM_POINTS * sizeof(double), cudaMemcpyHostToDevice));
	

	//Calcola la distanza tra i dati ed i cluster
	double old_error;
	double error = DBL_MAX;
	int cycle_counter = 0;

	clock_t begin = clock();
	do {
		
		cycle_counter++;
		old_error = error;
		error = 0;

		CHECK(cudaMemcpy(dev_centroids, host_centroids, k*DIM*sizeof(double), cudaMemcpyHostToDevice));
		CHECK(cudaMemset(dev_tmp_centroids, 0, k*DIM*sizeof(double)));
		CHECK(cudaMemset(dev_counts, 0, k*sizeof(int)));

		//O(kd)
		finding_closest<<<20,256>>>(dev_data_points, NUM_POINTS, DIM, dev_centroids, k, dev_labels, dev_min_distances, dev_tmp_centroids, dev_counts);
	
		CHECK(cudaDeviceSynchronize());
		
		CHECK(cudaMemcpy(host_min_distances, dev_min_distances, NUM_POINTS * sizeof(double), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(host_labels, dev_labels, NUM_POINTS * sizeof(int), cudaMemcpyDeviceToHost));
 		CHECK(cudaMemcpy(host_tmp_centroids, dev_tmp_centroids, k * DIM * sizeof(double), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(host_counts, dev_counts, k * sizeof(int), cudaMemcpyDeviceToHost));

		CHECK(cudaDeviceSynchronize());

		for(i=0;i<NUM_POINTS;i++){
			error += host_min_distances[i];
		}

		//O(n)
		
		// Calcolo dei nuovi centroidi
		double *tc;
		double *centroid;
		for(i=0; i<k; i++){
			centroid = &host_centroids[i * DIM];
			tc = &host_tmp_centroids[i * DIM];
			for(j=0; j<DIM; j++){
				if(host_counts[i]>0){
					centroid[j] = tc[j] / host_counts[i];
				}
			}
		}

		//O(kd)
		
	} while(fabs(error-old_error) > THRESHOLD);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLK_TCK;
	printf("%lf\n",time_spent );
	
	// Disegna il grafico con gnuplot
	plot(host_data_points, NUM_POINTS, DIM, host_labels, k);

	


	// liberazione memoria host
	free(host_labels);
	free(host_data_points);
	free(host_centroids);
	free(host_min_distances);
	free(host_tmp_centroids);
	free(host_counts);
	
	// Liberazione memoria device
	cudaFree(dev_min_distances);
	cudaFree(dev_data_points);
	cudaFree(dev_centroids);
	cudaFree(dev_labels);
	cudaFree(dev_tmp_centroids);
	cudaFree(dev_counts);

	return 0;
}


void plot(double *data_points, int n, int m, int *labels, int k){
	int i = 0;
	int j = 0;

	#define NUM_COMMANDS 2
	char * commandsForGnuplot[] = {"set title \"Parallel k-means\"", "plot 'data.temp' u 1:2:3:3 with labels tc palette"};
	
	FILE * temp = fopen("data/plot/data.temp", "w");

	FILE * gnuplotPipe = _popen ("gnuplot -persistent", "w");
	
	for(i=0; i < n;i++){
		double *tmp = &data_points[i*m];
		for(j=0;j<m;j++){
			fprintf(temp, "%lf ", tmp[j]); 
		}
		fprintf(temp, "%d\n",labels[i]);
	}

	for (i=0; i < NUM_COMMANDS; i++){
		fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]);
	}
}