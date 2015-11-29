/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

/**
 *
 * @author asus
 */
public class InputNeuron {
    private double input;
    private int id;
    
    public double getInput(){
        return input;
    }
    
    public int getId(){
        return id;
    }
    
    public void setInput(double _input){
        this.input = _input;
    }
    
    public void setId(int _id){
        this.id = _id;
    }
}
