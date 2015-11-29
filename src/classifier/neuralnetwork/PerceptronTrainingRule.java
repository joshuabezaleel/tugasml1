/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author asus
 */
public class PerceptronTrainingRule extends SingleLayerPerceptron {

    public PerceptronTrainingRule(int _maxIteration, List<Node> _listOfNode, double _learningRate, double _momentum) {
        super(_maxIteration, _listOfNode, _learningRate, _momentum);
    }   
    
    private double[] previousWeight;
    private double[] lastWeight;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.setInstances(data);
    }
    
}
