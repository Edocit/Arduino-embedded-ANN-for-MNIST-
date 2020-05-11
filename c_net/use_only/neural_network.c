

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "neural_network.h"



//PROTOTYPES

//FUNZIONI DI CARICAMENTO E DI DUMP
void dump_weights_and_biases(const float (*weights_matrix)[INPUT_SHAPE], const float *bias);
//CICLO DI ALLENAMENTO/PREDIZIONE
void normalize_image(float *vectorized_image);
void affine_transformation(const float (*mtx)[INPUT_SHAPE], float *vectorized_image, const float *bias_vect, float *result);
void cumulative_output_softmax(float *activations);
void predict(float *probs, int *class);
//WRAPPER delle 4 funzioni precedenti
void net_2_prediction(const float (*weights_mtx)[INPUT_SHAPE], const float *bias, float *input_image, float *result, int *predicted_label);

//VARIABILI GLOBALI DI RETE
float   input_image[FLATTEN_IMAGE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

extern const float weights_matrix[OUTPUT_SHAPE][INPUT_SHAPE];   //matrice dei pesi 
extern const float bias[OUTPUT_SHAPE];                          //bias dello strato di output
float   res[10];                                                //uscita della rete in termini di probabilità (dopo la softmax)
int     actual_label;                                           //etichetta predetta da paragonare a quella sopra per la backpropagation







void dump_weights_and_biases(const float (*weights_matrix)[INPUT_SHAPE], const float *bias){
    for(int i = 0; i < OUTPUT_SHAPE; i++){
        for(int j = 0; j < INPUT_SHAPE; j++){
            printf("%.10f, ", weights_matrix[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------\n");
     for(int i = 0; i < OUTPUT_SHAPE; i++){
         printf("%f, ", bias[i]);
     }
}




/* Trasformo il vettore di interi a 8 bits nel range normalizzato [0; 255] --> [0; 1] */
void normalize_image(float *vectorized_image){
    int i;

    for(i = 0; i < FLATTEN_IMAGE; i++){
        vectorized_image[i] /= MNIST_MAX_VAL;
    }
}




void affine_transformation(const float (*mtx)[INPUT_SHAPE], float *vectorized_image, const float *bias_vect, float *result){
    int i, k; 

    for(i = 0; i < OUTPUT_SHAPE; i++){
        for(k = 0; k < FLATTEN_IMAGE; k++){ 
            result[i] += ( mtx[i][k] * vectorized_image[k] );
        }

        result[i] += bias_vect[i];
    }
}




void cumulative_output_softmax(float *activations){
  float m = -INFINITY;

  for (size_t i = 0; i < LABELS; i++) {
    if (activations[i] > m) {
      m = activations[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < OUTPUT_SHAPE; i++) {
    sum += expf(activations[i] - m);
  }

  float offset = m + logf(sum);

  for (size_t i = 0; i < OUTPUT_SHAPE; i++) {
    activations[i] = expf(activations[i] - offset);
  }
}



void predict(float *probs, int *class){
    float max = 0;
    int i;

    for(i = 0; i < OUTPUT_SHAPE; i++){
        printf("%.4f, ", probs[i]);
        if(probs[i] > max){
            max = probs[i];
            *class = i;
        }
    }

    printf("Predicted number: %d\n", *class);

};




void net_2_prediction(const float (*weights_mtx)[INPUT_SHAPE], const float *bias, float *input_image, float *result, int *predicted_label){
    normalize_image(input_image);
    affine_transformation(weights_mtx, input_image, bias, result);
    cumulative_output_softmax(result);
    predict(result, predicted_label);
}





int main(){
    //dump_weights_and_biases(weights_matrix, bias);
    net_2_prediction(weights_matrix, bias, input_image, res, &actual_label);

    return 0;
}

