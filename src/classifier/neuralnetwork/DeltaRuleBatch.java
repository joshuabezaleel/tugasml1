package classifier.neuralnetwork;

import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author asus
 */
public class DeltaRuleBatch extends SingleLayerPerceptron {

    public DeltaRuleBatch(int _maxIteration, List<Node> _listOfNode, double _learningRate, double _momentum) {
        super(_maxIteration, _listOfNode, _learningRate, _momentum);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
