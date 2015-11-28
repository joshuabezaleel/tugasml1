/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author asus
 */
public class SingleLayerPerceptron extends Classifier {
    
    private int maxIteration;
    private List<Node> listOfNode;
    private double learningRate;
    private double momentum;
    private Instances instances;
    
    public int getMaxIteration() {
        return maxIteration;
    }
    
    public void setMaxIteration(int _maxIteration){
        this.maxIteration = _maxIteration;
    }
    
    public List<Node> getListOfNode(){
        return listOfNode;
    }
    
    public void setListOfNode(List<Node> _listOfNode){
        this.listOfNode = _listOfNode;
    }
    
    public double getLearningRate(){
        return learningRate;
    }
    
    public void setLearningRate(double _learningRate){
        this.learningRate = _learningRate;
    }
    
    public double getMomentum(){
        return momentum;
    }
    
    public void setMomentum(double _momentum){
        this.momentum = _momentum;
    }
    
    public Instances getInstances(){
        return instances;
    }
    
    public void setInstances(Instances _instances){
        this.instances = _instances;
    }
    
    public SingleLayerPerceptron(int _maxIteration, List<Node> _listOfNode, double _learningRate, double _momentum){
        this.maxIteration = _maxIteration;
        this.listOfNode = _listOfNode;
        this.learningRate = _learningRate;
        this.momentum = _momentum;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
