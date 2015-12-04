/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.io.Serializable;
import java.util.Random;

/**
 *
 * @author LINDA
 */
public class Node implements Serializable {
    private static final double BIAS = 1.0;
    private double value;
    private double error;
    private boolean useSigmoid;
    private double[] weights;
    private double[] deltaWeights;
    private double[] inputs;
    
    public Node() {
        useSigmoid = true;
    }
    
    public void setError(double err) {
        error = err;
    }
    
    public double getError() {
        return error;
    }
    
    public void setSigmoid(boolean use) {
        useSigmoid = use;
    }
    
    public double sigmoidFunction(double sum) {
        return 1/(1+Math.exp(-sum));
    }
    
    public void setInput(double[] input) {
        inputs = input.clone();
    }
    
    public void computeValue() {
        value = BIAS*weights[0];
        for(int i=0; i<inputs.length; i++) {
            value+=(inputs[i]*weights[i+1]);
        }
        if(useSigmoid)
            value = sigmoidFunction(value);
    }
    
    public double getValue() {
        return value;
    }
    
    public void initWeight(int nbNeuron, double w) {
        weights = new double[nbNeuron];
        deltaWeights = new double[nbNeuron];
        for(int i=0; i<nbNeuron; i++) {
            weights[i] = w;
            deltaWeights[i] = 0.0;
        }
    }
    
    public void randomizeWeight() {
        Random rand = new Random();
        for (int i=0; i<weights.length; i++) {
            weights[i] = rand.nextDouble();
        }
    }
    
    public void updateWeight(double learningRate) {
        for(int i=0; i<weights.length; i++) {
            if(i==0) {
                deltaWeights[i] = learningRate*error*BIAS;
            }
            else {
                deltaWeights[i] = learningRate*error*inputs[i-1];
            }
            weights[i] = weights[i]+deltaWeights[i];
        }
    }
    
    public void updateWeightWithPrevious(double learningRate, double momentumRate) {
        for(int i=0; i<weights.length; i++) {
            double prevDeltaWeight = deltaWeights[i];
            if(i==0) {
                deltaWeights[i] = learningRate*error*BIAS+prevDeltaWeight*momentumRate;
            }
            else {
                deltaWeights[i] = learningRate*error*inputs[i-1]+prevDeltaWeight*momentumRate;
            }
            weights[i] = weights[i]+deltaWeights[i];
        }
    }
    
    public double getSpecificWeight(int idxNeuron) {
        return weights[idxNeuron];
    }
    
    public double[] getWeights() {
        return weights;
    }
}
