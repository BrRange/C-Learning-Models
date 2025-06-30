#ifndef NNMH
#define NNMH

#include <stdlib.h>
#include <stdarg.h>

struct Model{
    size_t inputSize;
    float *bias;
    float *weight;
};
typedef struct Model Model;

struct TrainData{
    size_t dataSize, inputSize;
    float **data;
};
typedef struct TrainData TrainData;

Model newModel(size_t size){
    Model nnm = {size, malloc(4ull), malloc(4ull * size)};
    *nnm.bias = 1.f;
    for(size_t i = 0; i < size; i++)
        nnm.weight[i] = 1.f;
    return nnm;
}

float costModel(Model m, TrainData td){
    float res = 0.f;
    for(size_t i = 0; i < td.dataSize; i++){
        float partial = 0.f;
        for(size_t j = 0; j < m.inputSize; j++){
            partial += m.weight[j] * td.data[i][j] + *m.bias;
        }
        partial -= td.data[i][m.inputSize];
        res += partial * partial;
    }
    return res / td.dataSize;
}

void trainModel(Model m, TrainData td, float eps, float rate){
    float original, dcost;
    for(size_t i = 0; i < m.inputSize; i++){
        original = m.weight[i];
        m.weight[i] += eps;
        dcost = costModel(m, td);
        m.weight[i] = original;
        dcost -= costModel(m, td);
        dcost /= 2.f * eps;
        m.weight[i] -= rate * dcost;
    }
    original = *m.bias;
    *m.bias += eps;
    dcost = costModel(m, td);
    *m.bias = original;
    dcost -= costModel(m, td);
    dcost /= eps;
    *m.bias -= rate * dcost;
}

double outputModel(Model m, ...){
    double output = 0.;
    va_list args;
    va_start(args, m);
    for(size_t i = 0; i < m.inputSize; i++)
        output += va_arg(args, double) * m.weight[i];
    va_end(args);
    return output + *m.bias;
}

TrainData newTrainData(Model m, size_t size){
    TrainData td = {size, m.inputSize, malloc(4ull * size)};
    for(size_t i = 0; i < size; i++)
        td.data[i] = malloc(4ull * m.inputSize + 4ull);
    return td;
}

void fillTrainData(TrainData td, ...){
    va_list args;
    va_start(args, td);
    for(size_t i = 0; i < td.dataSize; i++)
    for(size_t j = 0; j <= td.inputSize; j++){
        td.data[i][j] = va_arg(args, double);
    }
    va_end(args);
}

#endif