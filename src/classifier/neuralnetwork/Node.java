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
    private double value;
    private List<Double> weight = new ArrayList<>();
    private List<Double> error = new ArrayList<>();
    
    public void setValue(double v) {
        value = v;
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
