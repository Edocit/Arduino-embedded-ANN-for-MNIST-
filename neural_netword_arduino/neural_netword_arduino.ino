
#include <avr/pgmspace.h>
#include <math.h>
#include "neural_network.h"


//PROTOTYPES
void normalize_image();
void affine_transformation();
void cumulative_output_softmax();
void predict();
//WRAPPER delle 4 funzioni precedenti
void net_prediction();
//Just to show the correct prediction 
void toggle(uint16_t blink_time_ms);



//VARIABILI GLOBALI DI RETE
extern PROGMEM const float weights_matrix[OUTPUT_SHAPE][INPUT_SHAPE];
extern PROGMEM const float bias[OUTPUT_SHAPE];
float res[10];                                                                      //uscita della rete in termini di probabilità (dopo la softmax)
int   actual_label;                                                                 //etichetta predetta da paragonare a quella sopra per la backpropagation

float input_image[FLATTEN_IMAGE] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 124, 176, 218, 226, 170, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 114, 230, 255, 255, 255, 255, 255, 236, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 168, 255, 255, 217, 151, 94, 90, 187, 255, 205, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 255, 235, 121, 17, 0, 0, 0, 4, 181, 255, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 153, 255, 87, 0, 0, 0, 0, 0, 0, 49, 248, 221, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 226, 227, 10, 0, 0, 0, 0, 0, 0, 0, 166, 255, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 255, 160, 0, 0, 0, 0, 0, 0, 0, 0, 36, 238, 235, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 255, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 255, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 134, 255, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 255, 170, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 252, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 209, 251, 49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 248, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 102, 255, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170, 249, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 255, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 149, 255, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 255, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 255, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 255, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 255, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127, 255, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 255, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 221, 241, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 214, 251, 70, 0, 0, 0, 0, 0, 0, 0, 0, 112, 255, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 255, 227, 48, 0, 0, 0, 0, 0, 0, 67, 236, 245, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 169, 255, 221, 45, 0, 0, 0, 0, 76, 236, 255, 139, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 178, 255, 238, 132, 46, 68, 170, 255, 255, 160, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 181, 255, 255, 253, 255, 255, 251, 145, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 97, 204, 249, 243, 172, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 39, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};






void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop(){
  //serial_net_input();
  
  net_prediction();
  while(1){ 
    if(actual_label == 0){ 
      toggle(250);
    }
    Serial.println(actual_label);
  }
}



//FUNCTIONS


void toggle(uint16_t blink_time_ms){
  digitalWrite(LED_BUILTIN,HIGH); 
  delay(blink_time_ms); 
  digitalWrite(LED_BUILTIN,LOW); 
  delay(blink_time_ms);  
}





/* Trasformo il vettore di interi a 8 bits nel range normalizzato [0; 255] --> [0; 1] */
void normalize_image(){
    int i;

    for(i = 0; i < FLATTEN_IMAGE; i++){
        input_image[i] /= MNIST_MAX_VAL;
    }
}




void affine_transformation(){
    int i, k; 

    for(i = 0; i < OUTPUT_SHAPE; i++){
        for(k = 0; k < FLATTEN_IMAGE; k++){ 
            res[i] += ( pgm_read_float(&weights_matrix[i][k]) * input_image[k] );
        }

        res[i] += pgm_read_float(&bias[i]);
        //Serial.print(String(res[i])+", ");
    }
    //Serial.println();
}




void cumulative_output_softmax(){
  float m = -INFINITY;
  for (int i = 0; i < LABELS; i++) {
    if (res[i] > m) {
      m = res[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < OUTPUT_SHAPE; i++) {
    sum += expf(res[i] - m);
  }

  float offset = m + logf(sum);

  for (int i = 0; i < OUTPUT_SHAPE; i++) {
    res[i] = expf(res[i] - offset);
  }

}



void predict(){
    float maxx = 0.0;
    int i;

    for(i = 0; i < OUTPUT_SHAPE; i++){
        if(res[i] > maxx){
            maxx = res[i];
            actual_label = i;
        }
    }
    
}




void net_prediction(){
    normalize_image();
    affine_transformation();
    cumulative_output_softmax();
    predict();
}
