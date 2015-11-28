/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

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
    
    public void addWeight(double w) {
        weight.add(w);
    }
    
    public double getOutput(int i) {
        return weight.get(i)*value;
    }
}
