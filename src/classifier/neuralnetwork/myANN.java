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
 * @author Rakhmatullah Yoga S, Linda Sekawati, Joshua Bezaleel Abednego
 */
public class myANN extends Classifier {
    private enum Function {
        SIGN, STEP, SIGMOID 
    }
    private Function func;
    private List<Node> input = new ArrayList<>();
    private List<Node> hiddenLayer = new ArrayList<>();
    private Node output;
    private double threshold;
    private double target;
    private int nbLayer;
    private int epoch;
    
    public double countInput(int idxLayer) {
        double outputSum = 0;
        for (Node inputNode : input) {
            outputSum += inputNode.getOutput(idxLayer);
        }
        return activationFunction(outputSum);
    }
    
    public double activationFunction(double sum) {
        double result;
        switch(func) {
            case SIGN:
                if(sum>=0)
                    result = 1;
                else
                    result = -1;
                break;
            case STEP:
                if(sum>=threshold)
                    result = 1;
                else
                    result = 0;
                break;
            case SIGMOID:
                result = 1/(1+Math.exp(-sum));
                break;
        }
        return 0;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
