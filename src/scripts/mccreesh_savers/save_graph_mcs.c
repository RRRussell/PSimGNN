/*
 The graph file format
 The graphs are stored in a compact binary format, one graph per file. The file is composed of 16 bit words, 
 which are represented using the so-called little-endian convention, i.e. the least significant byte of the word is stored first.
 
 Two different formats are used for labeled and unlabeled graphs. 
 In unlabeled graphs, the first word contains the number of nodes in the graph; this means that this format can deal with 
 graphs of up to 65535 nodes (however, actual graphs in the database are quite smaller, up to 1024 nodes). Then, for each node, 
 the file contains the list of edges coming out of the node itself. The list is represented by a word encoding its length, 
 followed by a word for each edge, representing the destination node of the edge. Node numeration is 0-based, so the first node 
 of the graph has index 0.The following C code shows how a graph file can be read into an adjacency matrix; the code assumes that 
 the input file has been opened using the binary mode.
 */


#include <stdio.h>
#define MAX 200
#define SAMPLE 20

/* WARNING: for simplicity this code does not check for End-Of-File! */
unsigned short read_word(FILE *in)
{
	unsigned char a[2];
	if (fread(a,1,2,in) != 2)
		printf("ERROR: could not decrypt binary graph.\n");
	return (int)a[0] | (((int)a[1]) << 8);

}

/* This function assumes that the adjacency matrix is statically allocated.
 * The return value is the number of nodes in the graph.
 */
int read_graph(FILE *in, int matrix[MAX][MAX], int node_attr[MAX], int edge_attr[MAX][MAX])
{
	int nvertices;
	int nedges;
	int target;
	int i, j;

	/* Read the number of nodes */
	nvertices = read_word(in);

	/* Clean-up nodes */
	for(i=0; i<nvertices; i++)
		node_attr[i]=0;

	/* Clean-up the matrix */
	for(i=0; i<nvertices; i++)
		for(j=0; j<nvertices; j++)
		{
			matrix[i][j]=0;
			edge_attr[i][j]=0;
		}

	/* Mccreesh's labeling scheme */
	int m = nvertices * 33 / 100;
	int p = 1;
	int k1 = 0;
	int k2 = 0;
	while (p < m && k1 < 16){
		p *= 2;
		k1 = k2;
		k2++;
	}

	/* Read the attributes of nodes */
	for(i=0; i<nvertices; i++)
	{
		// NOTE: this is slightly different in Mccreesh's paper
		// NOTE: Mccreesh's 'label' variables are actually 'features' or 'attributes'
		int attr = read_word(in) >> (16-k1);
		node_attr[i] |= attr;
	}

	/* Read the edges and edge attributes */
	for(i=0; i<nvertices; i++)
	{
		/* Read the number of edges coming out of node i */
		nedges = read_word(in); // equivalent to Mccreesh's 'len' variable

		/* For each edge out of node i... */
		for(j=0; j<nedges; j++)
		{
			/* Read the destination node of the edge */
			target = read_word(in);
			if (target > nvertices){
				printf("ERROR: connected with edge %d\n", target);
				return -1;
			}

			if (i == target)
				printf("WARNING: self-loop detected!\n");
			else
			{
				/* Insert edge in adjacency matrix */
				matrix[i][target] = 1;
				matrix[target][i] = 1;
				
				/* Insert edge attribute in edge attribute matrix */
				int attr = (read_word(in)>>(16-k1)) + 1;
				edge_attr[i][target] = attr;
				edge_attr[target][i] = attr;
			}
		}
	}
	return nvertices;
}

int main(int argc, int **argv){
	if( argc < 2 ){
		printf("ERROR: input file not specified!\n");
		return 1;
	}

	int adj_mat [MAX][MAX];
	int edge_attr [MAX][MAX];
	int node_attr [MAX];
	int nvertices;
	FILE *fp, *fp0, *fp1,*fp2, *fp3;
	fp = fopen(argv[1], "r");

	nvertices = read_graph(fp, adj_mat, node_attr, edge_attr);
	if (nvertices == -1)
		return 2;
	//printf("there are %d nodes\n", nvertices);

	// save the graph
	fclose(fp);
	fp0 = fopen("temp/n.txt", "a");
	fp1 = fopen("temp/adj_matrix.txt", "a");
	fp2 = fopen("temp/node_features.txt", "a");
	fp3 = fopen("temp/edge_features.txt", "a");

	/*
	// print for debugging purposes
	printf("====================================\nADJACENCY MATRIX:\n");

	for (i=0; i< SAMPLE; ++i){
		for (j = 0; j < SAMPLE; ++j){
			printf("%d", adj_mat[i][j]);
			if (j != SAMPLE-1){
				printf(",");
			}
		}
		printf("\n");
	}
	*/

	fprintf(fp0, "%d", nvertices);

	int i, j;
	for (i = 0; i < nvertices; ++i){
		for (j = 0; j < nvertices; ++j){
			fprintf(fp1, "%d", adj_mat[i][j]);
			fprintf(fp3, "%d", edge_attr[i][j]);
			if (j < nvertices-1){
				fprintf(fp1, ",");
				fprintf(fp3, ",");
			}
		}
		fprintf(fp1,"\n");
		fprintf(fp3,"\n");
	}
	
	for (i = 0; i < nvertices; ++i){
		fprintf(fp2, "%d", node_attr[i]);
		if (i != nvertices-1)
			fprintf(fp2, ",");
	}

	fclose(fp0);
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);

	printf("Graph Successfully Processed!\n");
	return 0;
}
