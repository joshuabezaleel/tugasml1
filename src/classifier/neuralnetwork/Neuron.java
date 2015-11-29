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
 * @author asus
 */
public class Neuron {
    private List<Double> previousWeight = new ArrayList<>();
    private List<Double> currentWeight = new ArrayList<>();
    private List<Double> deltaWeight = new ArrayList<>();
    private double value;
    private double error;
    
    public List<Double> getPreviousWeight(){
        return previousWeight;
    }
    
    public List<Double> getCurrentWeight(){
        return currentWeight;
    }
    
    public List<Double> getDeltaWeight(){
        return deltaWeight;
    }
    
    public void setDeltaWeight(int index, double weight){
        this.deltaWeight.set(index, weight);
    }
    
    public void setCurrentWeight(int index, double weight){
        this.currentWeight.set(index, weight);
    }
    
    public void setPreviousWeight(int index, double weight){
        this.previousWeight.set(index, weight);
    }
    
    public void initPreviousWeight(int nbNeuron, double w) {
        for(int i=0; i<nbNeuron; i++)
            previousWeight.add(w);
    }
    
    public void initCurrentWeight(int nbNeuron, double w) {
        for(int i=0; i<nbNeuron; i++)
            currentWeight.add(w);
    }
    
    public void initDeltaWeight(int nbNeuron, double w) {
        for(int i=0; i<nbNeuron; i++)
            deltaWeight.add(w);
    }
    
    public void randomizeCurrentWeight() {
        Random rand = new Random();
        for (int i=0; i<currentWeight.size(); i++) {
            currentWeight.set(i, rand.nextDouble());
        }
    }
    
    public void setCurrentWeightZero() {
        for (int i=0; i<currentWeight.size(); i++) {
            currentWeight.set(i, (double)0);
        }
    }
}
