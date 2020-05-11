

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <string.h>

//CARATTERISTICHE DELL' IMMAGINE (DEI DATI IN GENERALE)
#define IMAGE_WIDTH     28
#define IMAGE_HEIGHT    28
#define FLATTEN_IMAGE   ((IMAGE_WIDTH * IMAGE_HEIGHT))
#define INPUT_SHAPE     FLATTEN_IMAGE
#define OUTPUT_SHAPE    10
#define BIAS            10
#define LABELS          10
#define MNIST_MAX_VAL   255.0

//CARATTERISTICHE DEI FILE BINARI DEI DATASET IMMAGINI + ETICHETTE (PRESI DALLA DOCUMENTAZIONE SU KAGGLE)
#define IMAGE_MAGIC_NR  2051
#define LABEL_MAGIC_NR  2049
#define DATASET_OFFSET  16          //toglie in hex i primi 4 int32 addr{ 0x0000, 0x0004, 0x0008, 0x0012 }
#define TRAIN_IMAGES    60000
#define TRAIN_LABELS    TRAIN_IMAGES
#define TEST_IMAGES     10000
#define TEST_LABELS     TEST_IMAGES
#define IMAGE_HEADER    4
#define LABEL_HEADER    2

//IPERPARAMETRI DELLA RETE (MODIFICARE CON CURA E CON COGNIZIONE DI CAUSA)
#define LEARNING_RATE   0.02
#define BIAS_LERN_RATE  0.001
#define EPOCHS          1000
#define PATIENCE        20
#define MIN_DELTA       2e-05

//AUMENTANO LA LEGGIBILITÀ E AVVICINANO LA CHIAMATA DI FUNZIONE ALLE API DI KERAS/TENSORLOW (CHI HA LAVORATO IN PYTHON APPREZZERÀ)
enum early_stopping { ENABLED, DISABLED };
enum mode           { TRAIN, DISPATCH};


//PROTOTYPES

//FUNZIONI DI CARICAMENTO E DI DUMP
void load_weights_matrix_keras(float (*mtx)[INPUT_SHAPE]);
void load_bias_keras(float *bias);
void load_network_model(char *file_path_w, float (*weights_matrix)[INPUT_SHAPE], char *file_path_b, float *bias);
void dump_weights_and_biases(float (*weights_matrix)[INPUT_SHAPE], float *bias);
void initialize_weights_and_bias(float (*weights_matrix)[INPUT_SHAPE], float *bias);
int  swap_int_for_little_endian(int word);
void load_training_set_images(char *file_path, float (*trainset)[FLATTEN_IMAGE]);
void load_test_set_images(char *file_path, float (*testset)[FLATTEN_IMAGE]);
void load_labels(char *file_path, int *labels, int size);
//FUNZIONI DI LETTURA DEL DATASET E MANIPOLAZIONE DEL DATO
void normalize_image(float *vectorized_image);
void get_image_and_label_from_dataset(float *image, float (*dataset)[INPUT_SHAPE], int idx, int *labelset, int *exp_label);
//CICLO DI ALLENAMENTO PREDIZIONE
void matrix_by_vector_mul(float (*mtx)[INPUT_SHAPE], float* vectorized_image, float *result);
void affine_transformation(float (*mtx)[INPUT_SHAPE], float *vectorized_image, float *bias_vect, float *result);
void cumulative_output_softmax(float *activations);
void predict(float *probs, int *class);
//FUNZIONI DELLA FASE DI ALLENAMENTO (SOLO ALLENAMENTO)
void update_statistics(int actual_out, int expected_out, int *score);
void backpropagation(float (*mtx)[INPUT_SHAPE],float *input_image, int actual_out, int expected_out, float *bias, float *res, float *tot_loss);
void squared_error_loss_funcion(float *tot_loss, int samples);
void print_epoch_results(int idx, int *correct_classified, int training_set_dim, float *tot_loss, float *prev_tot_loss, int patience_cnt);
void early_stopping(float *tot_loss, float *prev_tot_loss, float min_delta, int patience, int *pat_cnt, int *epochs_ptr, int epochs, int enable);
void save_trained_model(char *file_path_w, float (*weights_matrix)[INPUT_SHAPE], char *file_path_b, float *bias);

//PERCORSI DEI FILES DA LEGGERE/SCRIVERE
char *train_images_p   = "dataset/train-images-idx3-ubyte";
char *train_labels_p   = "dataset/train-labels-idx1-ubyte";
char *test_images_p    = "dataset/t10k-images-idx3-ubyte";
char *test_labels_p    = "dataset/t10k-labels-idx1-ubyte";
char *weights_matrix_p = "net_data/weights_matrix.dat";
char *bias_p           = "net_data/biases.dat";


float   training_images[TRAIN_IMAGES][FLATTEN_IMAGE] = {{0},{0}};   //matrice di 60.000 immagini di 28x28=784 pixel l'una
float   test_images[TEST_IMAGES][FLATTEN_IMAGE]      = {{0},{0}};   //stessa cosa ma con 10.000 immagini
int     training_labels[TRAIN_IMAGES]                = {0};         //etichette della prima matrice quindi vettore di 60.000
int     test_labels[TEST_IMAGES]                     = {0};         //etichette della seconda matrice quindi vettore di 10.000

float   input_image[FLATTEN_IMAGE]                   = {0};         //vettore dell'immagine di input 784 <-- 28x28
float   weights_matrix[OUTPUT_SHAPE][INPUT_SHAPE]    = {{0},{0}};   //pesi (trasposta)... ogni nodo di input (784) è collegato ai 10 di output, quindi 10x784=7840 pesi allenabili 
float   bias[OUTPUT_SHAPE]                           = {0};         //i nodi non di input hanno i bias, quindi 10
float   res[10]                                      = {0};         //uscita della rete in termini di probabilità (dopo la softmax)
int     expected_label                               = 0;           //etichetta nota dal training set
int     actual_label                                 = 0;           //etichetta predetta da paragonare a quella sopra per la backpropagation
int     correct_classifications                      = 0;           //serve per calcolare l'accuracy score nella fase di training
float   tot_loss                                     = 0.0;         //valore di loss da minimizzare
float   prev_tot_loss                                = 0.0;         //valore di loss utilizzato per il delta della Early Stopping
int     patience_cnt                                 = 0;           //quando l'early stopping viene triggerata conta se il progresso della loss è minore per patience passi di fila
int     mode                                         = TRAIN;

void load_weights_matrix_keras(float (*mtx)[INPUT_SHAPE]){
    FILE *fd;
    char line[400];
    size_t len = 0;
    char *num;
    float num_buf[10];
    int i,k,j;

    j = 0;

    fd = fopen("Output_Weights.txt", "r");
    if(fd == NULL){ printf("Error"); exit(-1); }

    while (fgets(line, sizeof(line), fd)) {
        i = 0;
        while(i < 10){
            num = strtok(line,",");
            if(num == NULL){ break; }

            num_buf[i] = atof(num);
            i += 1;
        }

        for(i = 0; i < 10; i++){
            mtx[i][j] = num_buf[i];
        }
        j++;
    }
    fclose(fd);
}


void load_bias_keras(float *bias){
    FILE *fd;
    char line[400];
    size_t len = 0;
    char *num;
    float num_buf[10];
    int i;

    i = 0;

    fd = fopen("Output_Biases.txt", "r");
    if(fd == NULL){ printf("Error"); exit(-1); }

    while (fgets(line, sizeof(line), fd)) {
        bias[i] = atof(line);
        i++;
    }
    fclose(fd);    
}





void load_network_model(char *file_path_w, float (*weights_matrix)[INPUT_SHAPE], char *file_path_b, float *bias){
    int check;
    FILE *fd;

    fd = fopen(file_path_w, "rb");
    if(fd == NULL){ printf("Errore nel caricamento dei pesi da %s\n", file_path_w); fclose(fd); exit(-10); }

    check = fread(weights_matrix, sizeof(float), OUTPUT_SHAPE * INPUT_SHAPE, fd);

    if(check != (OUTPUT_SHAPE * INPUT_SHAPE) ){ 
        printf("Numero di pesi letto da %s non consistente con la matrice", file_path_w); 
        fclose(fd);
        exit(-11);
    }

    fclose(fd);

    fd = fopen(file_path_b, "rb");
    if(fd == NULL){ printf("Errore nel caricamento dei bias da %s\n", file_path_b); fclose(fd); exit(-12); }

    check = fread(bias, sizeof(float), OUTPUT_SHAPE, fd);

    if(check != OUTPUT_SHAPE){ 
        printf("Numero di bias letto da %s non consistente con il vettore", file_path_b); 
        fclose(fd);
        exit(-13);
    }

    fclose(fd);
}






void dump_weights_and_biases(float (*weights_matrix)[INPUT_SHAPE], float *bias){
    for(int i = 0; i < OUTPUT_SHAPE; i++){
        for(int j = 0; j < INPUT_SHAPE; j++){
            printf("%f ", weights_matrix[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------\n");
     for(int i = 0; i < OUTPUT_SHAPE; i++){
         printf("%f ", bias[i]);
     }
}




void initialize_weights_and_bias(float (*weights_matrix)[INPUT_SHAPE], float *bias){
    srand(time(NULL));

    for(int i = 0; i < OUTPUT_SHAPE; i++){
        for(int j = 0; j < INPUT_SHAPE; j++){
            weights_matrix[i][j] = (float)((rand() % 10000) / 10000.0);
        }

        bias[i] = 0;
    }
}




/* Converte il formato dell'intero da BIG endian 1 2 3 4 a LITTLE ENDIAN 4 3 2 1 */
int swap_int_for_little_endian(int word){
    int swapped =     ((word>>24)&0xff)       | // sposto il byte 3 al byte 0
                      ((word<<8)&0xff0000)    | // sposto il byte 1 al byte 2
                      ((word>>8)&0xff00)      | // sposto il byte 2 al byte 1
                      ((word<<24)&0xff000000);  // sposto il byte 0 al byte 3
    return swapped;
}





void load_training_set_images(char *file_path, float (*trainset)[FLATTEN_IMAGE]){
    int i, k, check;
    int32_t infos[4] = {0};
    int32_t values[4] = {IMAGE_MAGIC_NR, TRAIN_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT};
    uint8_t byte_data[FLATTEN_IMAGE] = {0};
    FILE *fd;

    fd = fopen(file_path, "rb");
    if(fd == NULL){ printf("Errore in trainig set"); exit(-1); }

    fread(infos, sizeof(int32_t), 4, fd); //salva a partire da infos 4 interi a 32 bit letti da fd

    for(i = 0; i < IMAGE_HEADER; i++){ 
        infos[i] = swap_int_for_little_endian(infos[i]); 
        if(infos[i] != values[i]){ printf("Errore nell' header"); exit(-2); }
    }

    for(i = 0; i < TRAIN_IMAGES; i++){
        check = fread(byte_data, sizeof(uint8_t), FLATTEN_IMAGE, fd);
        if(check != FLATTEN_IMAGE){ printf("Errore in lettura TS"); exit(-3); }

        for(k = 0; k < FLATTEN_IMAGE; k++){
            trainset[i][k] = (float)byte_data[k] / MNIST_MAX_VAL;  //il dato viene normalizzato dividendo per 255 che è il max su 8 bit
        }
    }

    fclose(fd);
}




void load_test_set_images(char *file_path, float (*testset)[FLATTEN_IMAGE]){
    int i, k, check;
    int32_t infos[IMAGE_HEADER] = {0};
    int32_t values[IMAGE_HEADER] = {IMAGE_MAGIC_NR, TEST_IMAGES, IMAGE_WIDTH, IMAGE_HEIGHT};
    uint8_t byte_data[FLATTEN_IMAGE] = {0};
    FILE *fd;

    fd = fopen(file_path, "rb");
    if(fd == NULL){ printf("Errore in test set"); exit(-4); }

    fread(infos, sizeof(int32_t), IMAGE_HEADER, fd); //salva a partire da infos 4 interi a 32 bit letti da fd

    for(i = 0; i < IMAGE_HEADER; i++){ 
        infos[i] = swap_int_for_little_endian(infos[i]); 
        if(infos[i] != values[i]){ printf("Errore nell' header"); exit(-5); }
    }

    for(i = 0; i < TEST_IMAGES; i++){
        check = fread(byte_data, sizeof(uint8_t), FLATTEN_IMAGE, fd);
        if(check != FLATTEN_IMAGE){ printf("Errore in lettura TS"); exit(-6); }

        for(k = 0; k < FLATTEN_IMAGE; k++){
            testset[i][k] = (float)byte_data[k] / MNIST_MAX_VAL;  //il dato viene normalizzato dividendo per 255 che è il max su 8 bit
        }
    }

    fclose(fd);
}



void load_labels(char *file_path, int *labels, int size){
    int i, check;
    int32_t infos[LABEL_HEADER] = {0};
    int32_t values[LABEL_HEADER];
    uint8_t byte_data[TRAIN_LABELS] = {0};
    FILE *fd;

    if(size == TRAIN_LABELS){ values[0] = LABEL_MAGIC_NR; values[1] = TRAIN_LABELS; }
    if(size == TEST_LABELS ){ values[0] = LABEL_MAGIC_NR; values[1] = TEST_LABELS;  }

    fd = fopen(file_path, "rb");
    if(fd == NULL){ printf("Errore in training labels"); exit(-5); }

    fread(infos, sizeof(int32_t), LABEL_HEADER, fd); //salva a partire da infos 4 interi a 32 bit letti da fd

    for(i = 0; i < LABEL_HEADER; i++){ 
        infos[i] = swap_int_for_little_endian(infos[i]); 
        if(infos[i] != values[i]){ printf("Errore nell' header"); exit(-6); }
    }

    check = fread(byte_data, sizeof(uint8_t), size, fd);
    if(check != size){ printf("Errore in lettura TS"); exit(-7); }

    for(i = 0; i < size; i++){ 
        labels[i] = (int)byte_data[i];
    }

    fclose(fd);    
}



void get_image_and_label_from_dataset(float *image, float (*dataset)[INPUT_SHAPE], int idx, int *labelset, int *exp_label){
    int i;

    for(i = 0; i < FLATTEN_IMAGE; i++){
        image[i] = dataset[idx][i];
    }

    *exp_label = labelset[idx];
}




/* Trasformo il vettore di interi a 8 bits nel range normalizzato [0; 255] --> [0; 1] */
void normalize_image(float *vectorized_image){
    int i;

    for(i = 0; i < FLATTEN_IMAGE; i++){
        vectorized_image[i] /= MNIST_MAX_VAL;
    }
}





void matrix_by_vector_mul(float (*mtx)[INPUT_SHAPE], float* vectorized_image, float *result){
    int i, j, k; 

    for(i = 0; i < OUTPUT_SHAPE; i++){
        for(k = 0; k < FLATTEN_IMAGE; k++){ 
            result[i] += ( mtx[i][k] * vectorized_image[k] );
        }
    }

} 



void affine_transformation(float (*mtx)[INPUT_SHAPE], float *vectorized_image, float *bias_vect, float *result){
    int i;

    matrix_by_vector_mul(mtx, vectorized_image, result);

    for(i = 0; i < OUTPUT_SHAPE; i++){
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
    //printf("Il totale --> %f\n", tot);
}



void predict(float *probs, int *class){
    int   idx = 0;
    float max = 0;
    int i;

    for(i = 0; i < OUTPUT_SHAPE; i++){
        //printf("%f ", probs[i]);
        if(probs[i] > max){
            max = probs[i];
            *class = i;
        }
    }
    //printf("\n");
    //printf("\nPrediction: %d ", *class);
};


void update_statistics(int actual_out, int expected_out, int *score){
    if(actual_label == expected_label){
        *score = *score + 1;
    }
}


/* Per applicare la backpropagation usiamo come funzione di errore la funzione "errore quadratico" *
 * definita come 0.5 * sum(expected_out - prob_out)^2                                              */
void backpropagation(float (*mtx)[INPUT_SHAPE],float *input_image, int actual_out, int expected_out, float *bias, float *res, float *tot_loss){
    int i, k; 
    float loss = 0.0;
    float err[OUTPUT_SHAPE] = {0};
    float loss_derivative[OUTPUT_SHAPE] = {0};
    float softmax_derivative[OUTPUT_SHAPE] = {0};
    float pred_out[LABELS] = {0};
    float real_out[LABELS] = {0};

    //faccio un efficientissimo OneHot Encoding sfruttando l'inizializzazione del vettore a 0 e piazzando a 1 solo l'indice desiderato
    pred_out[actual_out]   = 1;
    real_out[expected_out] = 1;

    //calcolo l'errore come la differenza tra il vettore categorico associato e le probabilita trovate componente per componente
    for(i = 0; i < OUTPUT_SHAPE; i++){
        err[i]                = real_out[i] - res[i];   //questo mi da le componenti della funzione errore
        loss_derivative[i]    = res[i] - real_out[i];   //questo mi da le derivate parziali del funzionale di costo (loss)
        softmax_derivative[i] = res[i] * (1 - res[i]);  //questo perchè differenziamo sempre rispetto alla variabile argomento della funzione
        loss += pow(err[i], 2);
    }
    
    loss /= 2;
    *tot_loss += loss;
    //applico la delta rule per la backpropagation
    for(i = 0; i < OUTPUT_SHAPE; i++){
        for(k = 0; k < FLATTEN_IMAGE; k++){ 
            mtx[i][k] -= LEARNING_RATE * loss_derivative[i] * softmax_derivative[i] * input_image[k];
        }
        bias[i] -= BIAS_LERN_RATE * loss_derivative[i] * softmax_derivative[i];
    }
}





void evaluate_test_set(float (*testset)[FLATTEN_IMAGE], int *test_labels){
    int k;

    for(k = 0; k < TEST_IMAGES; k++){
        get_image_and_label_from_dataset(input_image, test_images, k, test_labels, &expected_label);
        affine_transformation(weights_matrix, input_image, bias, res);
        cumulative_output_softmax(res);
        predict(res, &actual_label);
        update_statistics(actual_label, expected_label, &correct_classifications);
    }

    printf("\nTest set evaluation: %d / %d | %.2f%%\n", correct_classifications, TEST_LABELS, (float)(correct_classifications*100.0)/TEST_IMAGES);
}


/* La loss di fine epoca è data dalla media delle loss ottenute per ogni immagine del training set */
void squared_error_loss_funcion(float *tot_loss, int samples){
    *tot_loss /= samples; 
}




void print_epoch_results(int idx, int *correct_classified, int training_set_dim, float *tot_loss, float *prev_tot_loss, int patience_cnt){
    int   ep        = idx + 1;
    float acc       = (float)( (*correct_classified)*100.0 ) / training_set_dim;
    char *spaces = "";

    if(        ep < 10       ){   spaces = "   ";  }
    if(ep >= 10  && ep < 100 ){   spaces = "  ";   }
    if(ep >= 100 && ep < 1000){   spaces = " ";    }

    printf("%d%s) %d / %d | acc: %.2f%% | loss: %.6f | loss-delta: %.6f  | pat: %d\n", ep, spaces, *correct_classified, training_set_dim, acc, *tot_loss, *prev_tot_loss - *tot_loss, patience_cnt);
    *correct_classified = 0;
    *prev_tot_loss = *tot_loss;
    *tot_loss = 0.0;
}




/* Se la differenza tra i valori di loss è sotto una soglia minima (min_delta) stabilita da voi *
 * l'allenamento viene interrotto... questo è molto utile per permettere un allenamento lungo   *
 * quando non si ha idea del numero di epoche da dare al modello; viceversa permette di fermare *
 * un allenamento che si può ritenere concluso senza andare in overfitting o dover aspettare il *
 * numero di epoche rimanenti... CONSIGLIO: se abilitate questa modalità sovrastimate, anche in *
 * modo molto accentuato il numero di epoche di allenamento (non avete nulla da perderci ;)     */
void early_stopping(float *tot_loss, float *prev_tot_loss, float min_delta, int patience, int *pat_cnt, int *epochs_ptr, int epochs, int enable){
    if(enable == ENABLED && *epochs_ptr > 0){
        float delta = *prev_tot_loss - *tot_loss;

        if(delta < min_delta){
            *pat_cnt += 1;
        }else{
            *pat_cnt = 0;
        }

        if(*pat_cnt == patience){
            *epochs_ptr = epochs - 1;
            printf("\nEARLY STOPPING ATTIVATA\n\n");
        }
    }
}





void save_trained_model(char *file_path_w, float (*weights_matrix)[INPUT_SHAPE], char *file_path_b, float *bias){
    FILE *fd;
    int check;

    //Qui i pesi
    fd = fopen(file_path_w, "wb");
    if(fd == NULL){ printf("Errore in apertura del file binario %s in modalita wb", file_path_w); exit(-8); }

    check = fwrite(weights_matrix, sizeof(float), OUTPUT_SHAPE*INPUT_SHAPE, fd);

    if(check != OUTPUT_SHAPE*INPUT_SHAPE){ 
        printf("Dati in %s non consistenti con quelli richiesti", file_path_w);
        fclose(fd);
        exit(-9); 
    }

    printf("\n\nScritti correttamente %d valori su %s\n", check, file_path_w);
    fclose(fd);

    //Ora i bias
    fd = fopen(file_path_b, "wb");
    if(fd == NULL){ printf("Errore in apertura del file binario %s in modalita wb", file_path_b); exit(-9); }

    check = fwrite(bias, sizeof(float), OUTPUT_SHAPE, fd);

    if(check != OUTPUT_SHAPE){ 
        printf("Dati in %s non consistenti con quelli richiesti", file_path_b);
        fclose(fd);
        exit(-9); 
    }

    printf("Scritti correttamente %d valori su %s\n", check, file_path_b);
    fclose(fd);

    printf("RETE SALVATA\n\n");
}




int main(){
    int i,k;

    if(mode == TRAIN){
        load_training_set_images(train_images_p, training_images);
        load_test_set_images(test_images_p, test_images);
        load_labels(train_labels_p, training_labels, TRAIN_LABELS);
        load_labels(test_labels_p, test_labels, TEST_LABELS);
        initialize_weights_and_bias(weights_matrix, bias);

        for(i = 0; i < EPOCHS; i++){
            for(k = 0; k < TRAIN_IMAGES; k++){
                get_image_and_label_from_dataset(input_image, training_images, k, training_labels, &expected_label);
                affine_transformation(weights_matrix, input_image, bias, res);
                cumulative_output_softmax(res);
                predict(res, &actual_label);
                update_statistics(actual_label, expected_label, &correct_classifications);
                backpropagation(weights_matrix, input_image, actual_label, expected_label, bias, res, &tot_loss);
            }
            squared_error_loss_funcion(&tot_loss, TRAIN_IMAGES);
            early_stopping(&tot_loss, &prev_tot_loss, MIN_DELTA, PATIENCE, &patience_cnt, &i, EPOCHS, ENABLED);
            print_epoch_results(i, &correct_classifications, TRAIN_IMAGES, &tot_loss, &prev_tot_loss, patience_cnt);
        }

        evaluate_test_set(test_images, test_labels);
        save_trained_model(weights_matrix_p, weights_matrix, bias_p, bias);
    }else{
        load_network_model(weights_matrix_p, weights_matrix, bias_p, bias);

        float a[FLATTEN_IMAGE] = { 0 }; //metti qual la tua immagine codificata
        
        normalize_image(a);
        affine_transformation(weights_matrix, a, bias, res);
        cumulative_output_softmax(res);
        predict(res, &actual_label);
        printf("Predicted number: %d\n", actual_label);
    }

    return 0;
}