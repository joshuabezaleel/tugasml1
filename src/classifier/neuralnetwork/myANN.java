/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Rakhmatullah Yoga S
 */
public class myANN extends Classifier {
    private List<Node> input = new ArrayList<>();
    private List<Node> hiddenLayer = new ArrayList<>();
    private Node output;
    
    
    private float deltaW;                           //nilai delta dari weight
    private ArrayList listInput = new ArrayList();       //list fitur-fitur pada satu input
    private ArrayList listNodeLayer  = new ArrayList();  //list node pada sebuah layer
    private ArrayList listInstance = new ArrayList();    //list instance
    private ArrayList targetVal  = new ArrayList();      //list nilai target
    private int nbClass;
    
    
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
