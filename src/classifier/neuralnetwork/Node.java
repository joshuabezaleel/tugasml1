/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author LINDA
 */
public class Node {
    private static final double BIAS = 1.0;
    private double value;
    private boolean useSigmoid;
    private List<Double> weight = new ArrayList<>();
    private List<Double> input = new ArrayList<>();
    private List<Double> error = new ArrayList<>();
    
    public Node() {
        useSigmoid = true;
    }
    
    public void setSigmoid(boolean use) {
        useSigmoid = use;
    }
    
    public double sigmoidFunction(double sum) {
        return 1/(1+Math.exp(-sum));
    }
    
    public void setInput(double[] input) {
        for (double j : input) {
            this.input.add(j);
        }
    }
    
    public void computeValue() {
        value = BIAS*weight.get(0);
        for(int i=0; i<input.size(); i++) {
            value+=(input.get(i)*weight.get(i+1));
        }
        if(useSigmoid)
            value = sigmoidFunction(value);
    }
    
    public double getValue() {
        return value;
    }
    
    public void initWeight(int nbNeuron, double w) {
        for(int i=0; i<nbNeuron; i++)
            weight.add(w);
    }
    
    public void randomizeWeight() {
        Random rand = new Random();
        for (int i=0; i<weight.size(); i++) {
            weight.set(i, rand.nextDouble());
        }
    }
    
    public double getOutput(int i) {
        return weight.get(i)*value;
    }
}
